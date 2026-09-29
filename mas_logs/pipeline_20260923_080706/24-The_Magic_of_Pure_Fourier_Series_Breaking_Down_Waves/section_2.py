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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Language of Orthogonality", [
            "Periodic functions repeat their values regularly.",
            "Sine and cosine waves are mathematically orthogonal.",
            "This allows us to find individual frequency components."
        ])

        # Define elements
        v1 = Vector([1, 0, 0], color="#00FF00")
        v2 = Vector([0, 1, 0], color="#00FF00")
        label_v1 = Text("v1", font_size=24, color=WHITE)
        label_v2 = Text("v2", font_size=24, color=WHITE)
        
        v3 = Vector([0.7, 0.7, 0], color="#FF0000")
        proj_line = DashedLine(v3.get_end(), [0.7, 0, 0], color="#FF0000")
        proj_line2 = DashedLine(v3.get_end(), [0, 0.7, 0], color="#FF0000")
        
        vector_group = VGroup(v1, v2, v3, proj_line, proj_line2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.place_at_grid(VGroup(v1, v2), "C3")
        self.place_at_grid(label_v1, "C5", scale_factor=0.9)
        self.place_at_grid(label_v2, "A3", scale_factor=0.9)
        self.play(Create(v1), Create(v2), Write(label_v1), Write(label_v2))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.place_at_grid(v3, "B4", scale_factor=1.0)
        self.place_in_area(vector_group, "B3", "D5", scale_factor=0.8)
        self.play(GrowArrow(v3))
        self.play(Create(proj_line), Create(proj_line2))
        self.wait(2)
