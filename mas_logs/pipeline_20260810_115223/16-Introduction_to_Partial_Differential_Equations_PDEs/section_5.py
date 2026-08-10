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
        self.setup_layout("Applications and Summary", [
            "PDEs model complex real-world phenomena.",
            "From weather patterns to financial markets.",
            "Modeling, constraints, and solving define PDEs."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        thermometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thermometer.svg")
        heat_map = VGroup(Square(side_length=2, fill_opacity=0.6, color=YELLOW), thermometer)
        self.place_in_area(heat_map, 'B2', 'C4', scale_factor=0.7)
        self.play(Create(heat_map))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        satellite = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg")
        weather_icon = VGroup(Dot(radius=0.5, color=BLUE_B), satellite)
        self.place_at_grid(weather_icon, 'B6', scale_factor=0.8)
        self.play(FadeIn(weather_icon))
        self.play(weather_icon.animate.shift(DOWN * 0.5))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ORANGE)
        checklist = VGroup(
            Text("1. Model", font_size=20),
            Text("2. Constraints", font_size=20),
            Text("3. Solve", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(checklist, 'D5', 'F6', scale_factor=0.7)
        self.play(Write(checklist))
        self.wait(2)
