from manim import *
import os

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
        self.setup_layout("Prerequisite: Changing the 'Lens' of Distance", [
            "Numbers are small if divisible by two.",
            "Higher powers of two mean closer proximity.",
            "We use a tree to visualize this."
        ])
        
        # Elements
        CYAN_COLOR = "#00FFFF"
        number_line = NumberLine(x_range=[0, 10, 1], length=6, include_numbers=True)
        point_a = Dot(color=CYAN_COLOR).move_to(number_line.n2p(2))
        point_b = Dot(color=CYAN_COLOR).move_to(number_line.n2p(8))
        label_a = Text("A=2", font_size=20, color=CYAN_COLOR).next_to(point_a, UP)
        label_b = Text("B=8", font_size=20, color=CYAN_COLOR).next_to(point_b, UP)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg"
        if os.path.exists(asset_path):
            lens_icon = SVGMobject(asset_path, color=CYAN_COLOR)
        else:
            lens_icon = Dot(color=CYAN_COLOR)

        # === Animation for Lecture Line 1 ===
        # Using grid position for number_line as per issue 38 (D2-D5)
        self.place_in_area(number_line, 'D2', 'D5', scale_factor=1.0)
        self.play(Create(number_line), run_time=1)
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.add(point_a, point_b, label_a, label_b)
        self.play(FadeIn(point_a, point_b, label_a, label_b), run_time=1)
        self.lecture[1].set_color(CYAN_COLOR)

        # === Animation for Lecture Line 3 ===
        # Using grid position for lens_icon as per issue 37 (B3)
        self.place_at_grid(lens_icon, "B3", scale_factor=0.6)
        self.play(FadeIn(lens_icon), run_time=1)
        self.play(
            point_a.animate.set_color("#FF4500"),
            point_b.animate.set_color("#FF4500"),
            label_a.animate.set_color("#FF4500"),
            label_b.animate.set_color("#FF4500"),
            run_time=2
        )
        self.lecture[2].set_color("#FF4500")
        self.wait(1)
