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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Warm-up: Conditional Probability", [
            "Conditional probability updates our beliefs.", 
            "We narrow down possibilities with new information.", 
            "The 'universe' shrinks to a subset."
        ])
        
        # Define visual objects
        universe = Rectangle(width=4, height=4, color=WHITE)
        subset_a = Rectangle(width=2, height=2, color="#0000FF", fill_opacity=0.5)
        subset_b = Rectangle(width=2, height=2, color="#00FF00", fill_opacity=0.5)
        
        # Positioning updated per feedback
        self.place_in_area(universe, "A4", "C6", scale_factor=0.6)
        self.place_in_area(subset_a, "D1", "E2", scale_factor=0.5)
        self.place_in_area(subset_b, "D4", "E5", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(Create(universe), FadeIn(subset_a), FadeIn(subset_b))
        self.play(self.lecture[0].animate.set_color("#3399FF"))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF33"))
        self.play(subset_a.animate.set_stroke(width=6), subset_b.animate.set_stroke(width=1))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(FadeOut(subset_b), subset_a.animate.move_to(self.grid["C3"]))
