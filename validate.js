const { z } = require('zod');

// Define the schema for a valid HR query
const hrQuerySchema = z.object({
  body: z.object({
    question: z.string()
      .trim()
      .min(10, "Your question must be at least 10 characters long.")
      .max(500, "Your question is too long (max 500 characters).")
      // Basic security check: disallow common injection characters
      .refine((val) => !/[<>/{}]/.test(val), {
        message: "Question contains prohibited special characters."
      })
  })
});

// Middleware function to execute the validation
const validate = (schema) => (req, res, next) => {
  const result = schema.safeParse({ body: req.body });
  
  if (!result.success) {
    return res.status(400).json({
      error: result.error.errors[0].message
    });
  }
  next();
};

module.exports = { hrQuerySchema, validate };