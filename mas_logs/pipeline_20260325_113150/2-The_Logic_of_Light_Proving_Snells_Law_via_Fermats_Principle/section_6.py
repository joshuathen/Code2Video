from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section6Scene(TeachingScene):
    def construct(self):
        # 1. Setup the layout with the lecture lines
        # Required format for lecture lines as per instructions
        lecture_lines = [
            "We start with the ratio of sines over velocities.",
            "Replace velocities with speed of light and refractive indices.",
            "This yields the final form of Snell's Law."
        ]
        self.setup_layout("The Final Derivation: Snell's Law", lecture_lines)

        # Color definitions for visual matching with lecture lines
        color_step1 = WHITE
        color_step2 = TEAL
        color_step3 = "#FFD700"  # Gold

        # === Animation for Lecture Line 1 ===
        # Equation: sin(θ₁) / v₁ = sin(θ₂) / v₂
        self.play(self.lecture[0].animate.set_color(color_step1))
        # Resolution of Issue 45: eq1 at scale 1.0 (down from 1.1)
        eq1 = Text("sin(θ₁) / v₁ = sin(θ₂) / v₂", color=color_step1, font_size=28)
        self.place_in_area(eq1, "A2", "B5", scale_factor=1.0)
        self.play(Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Equation: sin(θ₁) / (c/n₁) = sin(θ₂) / (c/n₂)
        self.play(self.lecture[1].animate.set_color(color_step2))
        # Resolution of Issue 44: Scale 1.0 (down from 1.1) and using full width (cols 1-6)
        # Resolution of Issue 46: Moved to rows D-E to leave row C as a vertical buffer
        eq2 = Text("sin(θ₁) / (c/n₁) = sin(θ₂) / (c/n₂)", color=color_step2, font_size=28)
        self.place_in_area(eq2, "D1", "E6", scale_factor=1.0)
        self.play(ReplacementTransform(eq1.copy(), eq2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Equation: n₁ sin(θ₁) = n₂ sin(θ₂) inside a golden box
        self.play(self.lecture[2].animate.set_color(color_step3))
        eq3 = Text("n₁ sin(θ₁) = n₂ sin(θ₂)", color=color_step3, font_size=32)
        # Position at the very bottom area F1 to F6
        self.place_in_area(eq3, "F1", "F6", scale_factor=1.0)
        
        # Create a gold box around the final result
        box = SurroundingRectangle(eq3, color=color_step3, buff=0.15)
        
        self.play(
            ReplacementTransform(eq2.copy(), eq3),
            Create(box)
        )
        self.wait(3)
