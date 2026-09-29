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
        self.setup_layout("Visualizing the Solution: Slope Fields", [
            "ODEs define a slope field.",
            "Think of this as directional winds.",
            "Each arrow shows the local trend."
        ])
        
        # Create grid elements once
        grid_points = [self.grid[f"{row}{col}"] for row in ["B", "C", "D", "E", "F"] for col in ["2", "3", "4", "5", "6"]]
        dots = VGroup(*[Dot(pos, color=WHITE, radius=0.04) for pos in grid_points])
        
        arrows = VGroup(*[Arrow(start=pos - np.array([0.1, 0.05, 0]), 
                                end=pos + np.array([0.1, 0.05, 0]), 
                                color="#00FFFF", buff=0, tip_length=0.1) 
                          for pos in grid_points])

        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        flag = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/flag.svg")
        
        # Labels
        wind_field_label = Text("Wind Field", font_size=20, color="#00FFFF")

        # === Animation for Lecture Line 1: Generate a grid of dots in #FFFFFF ===
        self.play(Create(dots))
        self.lecture[0].set_color("#00FFFF")
        self.wait(0.5)
        
        # === Animation for Lecture Line 2: Transform dots into small directional arrows ===
        # Place Compass asset
        self.place_at_grid(compass, 'A2', scale_factor=0.3)
        self.place_at_grid(wind_field_label, 'A3', scale_factor=0.6)
        
        self.play(ReplacementTransform(dots, arrows), FadeIn(compass), Write(wind_field_label))
        self.lecture[1].set_color("#00FFFF")
        self.wait(0.5)
        
        # === Animation for Lecture Line 3: Pulse the arrows to emphasize local trend ===
        # Place Flag asset
        self.place_at_grid(flag, 'A4', scale_factor=0.3)
        self.play(FadeIn(flag))
        
        self.play(arrows.animate.set_color("#FF0000").scale(1.2), run_time=1)
        self.play(arrows.animate.set_color("#00FFFF").scale(1/1.2), run_time=1)
        self.lecture[2].set_color("#FF0000")
        self.wait(1)
