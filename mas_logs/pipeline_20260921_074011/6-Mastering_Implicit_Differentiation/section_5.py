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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary & Quick Check", [
            "Chain rule is key for implicit differentiation.",
            "Remember to always include dy/dx.",
            "Verify every y term derivative carefully."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Fix 32/39: Title at A3
        title = Text("Summary: Implicit Differentiation", color="#FFFFFF", font_size=32)
        self.place_at_grid(title, 'A3', scale_factor=0.9)
        self.play(Write(title))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        # Fix 32/39: Checklist in B2-D4
        checklist = VGroup(
            Text("1. Differentiate", font_size=24, color="#00FFFF"),
            Text("2. Group dy/dx", font_size=24, color="#00FFFF"),
            Text("3. Solve for dy/dx", font_size=24, color="#00FFFF")
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        self.place_in_area(checklist, 'B2', 'D4', scale_factor=0.8)
        self.play(FadeIn(checklist))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # === Animation for Lecture Line 3 ===
        # Fix 33/39: Success at E3
        success = Text("SUCCESS!", color="#00FF00", font_size=40)
        self.place_at_grid(success, 'E3', scale_factor=0.8)
        self.play(FadeIn(success))
        self.play(success.animate.scale(1.2), run_time=0.5)
        self.play(success.animate.scale(1/1.2), run_time=0.5)
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        self.wait(2)
