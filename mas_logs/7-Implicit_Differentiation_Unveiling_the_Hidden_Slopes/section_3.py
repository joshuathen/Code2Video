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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout(
            "The Step-by-Step Algorithm",
            [
                "First, differentiate both sides with respect to x.",
                "Apply the Chain Rule whenever you encounter y.",
                "The derivative of x squared is simply 2x.",
                "The derivative of y squared becomes 2y dy/dx.",
                "Finally, isolate dy/dx to find the slope."
            ]
        )

        # Define colors for synchronization
        COLOR_STEP1 = "#FFFFFF"  # White
        COLOR_STEP2 = "#00FF00"  # Green
        COLOR_STEP3 = "#FFD700"  # Gold

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_STEP1))
        
        # Issue 39 fix: Wider area and slightly smaller scale
        label1 = Text("Step 1: Differentiate both sides", font_size=24, color=COLOR_STEP1)
        self.place_in_area(label1, 'A1', 'A6', scale_factor=0.8)
        
        eq1 = Text("d/dx (x² + y²) = d/dx (25)", font_size=24, color=COLOR_STEP1)
        self.place_in_area(eq1, "B2", "B5")
        
        self.play(Write(label1), Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_STEP2))
        
        label2 = Text("Step 2: Chain Rule", font_size=24, color=COLOR_STEP2)
        self.place_in_area(label2, "C2", "C5")
        
        self.play(Write(label2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_STEP1))
        
        # Prepare the differentiated equation using a VGroup of Text to maintain indexing
        eq2 = VGroup(
            Text("2x", font_size=36),           # index 0
            Text(" + ", font_size=36),          # index 1
            Text("2y dy/dx", font_size=36),     # index 2
            Text(" = ", font_size=36),          # index 3
            Text("0", font_size=36)             # index 4
        ).arrange(RIGHT, buff=0.1)
        
        # Issue 38 fix: Wider area and scale adjustment
        self.place_in_area(eq2, 'D1', 'D6', scale_factor=0.8)
        
        # Color coordination
        eq2[0].set_color(COLOR_STEP1) # 2x
        eq2[2].set_color(COLOR_STEP2) # 2y dy/dx
        
        # Transform the first equation into the differentiated one
        self.play(ReplacementTransform(eq1.copy(), eq2))
        self.play(Indicate(eq2[0], color=COLOR_STEP1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_STEP2))
        
        # Highlight the y-derivative component
        self.play(Indicate(eq2[2], color=COLOR_STEP2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_STEP3))
        
        # Issue 37 fixes: Repositioning to avoid overlap and bottom edge clipping
        label3 = Text("Step 3: Isolate", font_size=24, color=COLOR_STEP3)
        self.place_in_area(label3, 'E1', 'E3', scale_factor=0.7)
        
        eq3 = Text("2y dy/dx = -2x", font_size=24, color=COLOR_STEP3)
        self.place_in_area(eq3, 'E4', 'E6', scale_factor=0.7)
        
        eq4 = Text("dy/dx = -x/y", font_size=24, color=COLOR_STEP3)
        self.place_in_area(eq4, 'F2', 'F5', scale_factor=1.0)
        
        self.play(Write(label3))
        # Step-by-step isolation
        self.play(ReplacementTransform(eq2.copy(), eq3))
        self.wait(1)
        self.play(ReplacementTransform(eq3, eq4))
        self.wait(2)
