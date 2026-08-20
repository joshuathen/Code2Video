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
        self.setup_layout("Linear Dependence: The Redundant Vector", [
            "Linear dependence occurs with redundant vectors.",
            "One vector as a combination of others.",
            "Redundant vectors add no new directional coverage."
        ])
        
        # Define origin asset and vectors
        origin_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/origin.svg")
        self.place_at_grid(origin_asset, 'C4', scale_factor=0.5)
        
        v1 = Arrow(start=origin_asset.get_center(), end=self.grid['B4'], color="#FF5733")
        v2 = Arrow(start=origin_asset.get_center(), end=self.grid['C6'], color="#33FF57")
        v3 = Arrow(start=origin_asset.get_center(), end=self.grid['B6'], color="#3357FF")
        
        vector_group = VGroup(v1, v2, v3)
        self.place_in_area(vector_group, 'B4', 'E6', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"), Create(origin_asset), Create(v1), Create(v2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        v3_label = MathTex("v_3 = v_1 + v_2", color="#3357FF").scale(0.7)
        self.place_in_area(v3_label, 'D3', 'E4', scale_factor=0.8)
        
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#3357FF"), Create(v3))
        self.play(Write(v3_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FF0000"), Indicate(v3))
        self.play(FadeOut(v3), FadeOut(v3_label))
        self.wait(2)
