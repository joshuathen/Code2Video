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
        self.setup_layout("The Foundation: Pseudo-Random IDs", [
            "Devices use rotating Ephemeral IDs.", 
            "IDs are derived from daily keys.", 
            "IDs change frequently for privacy."
        ])
        
        # Create elements
        key = Text("Daily Key", font_size=24, color=WHITE)
        self.place_at_grid(key, 'B2', scale_factor=0.8)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg
        smartphone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg", color=WHITE)
        id_text1 = Text("EphID-1", font_size=20)
        id_group1 = VGroup(smartphone, id_text1).arrange(DOWN)
        self.place_at_grid(id_group1, 'D2', scale_factor=0.5)
        
        arrow = Arrow(start=key.get_right(), end=id_group1.get_top(), color=WHITE)
        self.add(key, id_group1, arrow)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.play(
            Rotate(id_group1, angle=2*PI),
            smartphone.animate.set_color("#00FF00"),
            id_text1.animate.set_color("#00FF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        path_line = Line(start=key.get_bottom(), end=id_group1.get_top(), color="#00FFFF")
        self.play(Create(path_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/smartwatch.svg
        smartwatch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartwatch.svg", color="#FF00FF")
        id_text2 = Text("EphID-2", font_size=20, color="#FF00FF")
        id_group2 = VGroup(smartwatch, id_text2).arrange(DOWN)
        self.place_at_grid(id_group2, 'D2', scale_factor=0.5)
        
        self.play(
            FadeOut(id_group1),
            FadeIn(id_group2),
            path_line.animate.set_color("#FF00FF")
        )
        self.wait(2)
