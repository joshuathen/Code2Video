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
        self.setup_layout("Prerequisite Review: From Arrows to Objects", [
            "Vectors are not just arrows.",
            "They include polynomials and functions.",
            "Objects treated as points in space."
        ])
        
        # Objects
        obj_a = Dot(color=WHITE)
        obj_b = Dot(color=WHITE)
        
        # Adjusting positions based on VideoCritic feedback
        self.place_at_grid(obj_a, "C3", scale_factor=1.0)
        self.place_at_grid(obj_b, "E3", scale_factor=1.0)
        
        arrow = Arrow(start=self.grid["C3"], end=self.grid["E3"], color=WHITE)
        morphism_label = Text("Morphism", font_size=20, color=WHITE)
        objects_label = Text("Objects", font_size=24, color=WHITE)

        # Place labels based on VideoCritic feedback
        self.place_at_grid(objects_label, "B3", scale_factor=0.8)
        self.place_at_grid(morphism_label, "D3", scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        self.play(FadeIn(objects_label))
        self.play(Create(arrow))
        self.play(Write(morphism_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(arrow.animate.set_color("#FF00FF"), morphism_label.animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(Indicate(obj_a, color="#00FFFF"), Indicate(obj_b, color="#00FFFF"))
        self.wait(2)
