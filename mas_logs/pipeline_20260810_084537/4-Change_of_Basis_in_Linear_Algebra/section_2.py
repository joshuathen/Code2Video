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
        self.setup_layout("The Problem: Changing Perspectives", [
            "Now introduce a new, tilted basis system.",
            "The point remains stationary in space.",
            "Coordinates change to describe the same location."
        ])
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        # Basis 1
        axes1 = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True})
        vec1 = Vector([1, 1], color="#00FFFF")
        group1 = VGroup(axes1, vec1, compass).scale(0.5)
        
        # Basis 2
        axes2 = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True}).rotate(45*DEGREES)
        vec2 = Vector([1, 1], color="#FF00FF").rotate(45*DEGREES)
        group2 = VGroup(axes2, vec2, protractor).scale(0.5)
        
        static_point = Dot(color=YELLOW)
        coords_text = Text("(1, 1) -> (sqrt(2), 0)", font_size=20, color=YELLOW)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.place_in_area(group1, 'C2', 'E5', scale_factor=1.0)
        self.play(Create(group1))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        self.place_at_grid(static_point, 'D3', scale_factor=0.6)
        self.play(FadeIn(static_point))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.place_at_grid(coords_text, 'F3', scale_factor=0.5)
        self.play(Transform(group1, group2), FadeIn(coords_text))
        self.wait(1)
