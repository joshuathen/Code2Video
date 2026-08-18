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
        self.setup_layout("Summary and Connection", [
            "Det > 1 expands area, < 1 contracts.", 
            "Det = 0 indicates a total dimensional collapse.", 
            "Negative values always imply an orientation flip."
        ])
        
        # Assets
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg").set_color("#FFFFFF")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg").set_color("#FFFF00")
        
        # Shape
        square = Square(side_length=1.5, color=BLUE, fill_opacity=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), run_time=1)
        self.place_in_area(map_icon, 'A4', 'C6', scale_factor=0.5)
        self.place_in_area(square, 'A4', 'C6', scale_factor=0.5)
        self.play(FadeIn(map_icon), FadeIn(square))
        self.play(square.animate.scale(1.5), run_time=2)
        self.play(square.animate.scale(1/2.25), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"), run_time=1)
        self.place_at_grid(square, 'E3', scale_factor=0.6)
        self.play(square.animate.stretch(0, 0), run_time=2)
        self.play(square.animate.stretch(1, 0), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"), run_time=1)
        self.place_in_area(protractor, 'D3', 'F5', scale_factor=0.8)
        self.place_in_area(square, 'D3', 'F5', scale_factor=0.8)
        self.play(FadeIn(protractor), FadeIn(square))
        self.play(Rotate(square, angle=PI, axis=RIGHT), run_time=2)
        self.wait(2)
