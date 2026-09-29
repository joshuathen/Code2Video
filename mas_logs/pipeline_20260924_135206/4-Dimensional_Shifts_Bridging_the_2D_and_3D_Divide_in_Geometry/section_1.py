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
        self.setup_layout("The Flatland Limitation", [
            "Dimensions define our degree of movement.",
            "Flatlanders perceive only length and width.",
            "Depth remains hidden to a 2D world."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        plane = Square(side_length=4, color="#FFFFFF", fill_opacity=0.2)
        self.place_in_area(plane, 'A2', 'F5', scale_factor=0.9)
        self.play(FadeIn(plane))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#FF5733")
        flatlander = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg", color="#FF5733")
        self.place_at_grid(flatlander, 'B3', scale_factor=0.6)
        self.play(FadeIn(flatlander))
        self.play(flatlander.animate.shift(RIGHT * 1.5))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#33FF57")
        
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color="#33FF57")
        self.place_at_grid(circle, 'E4', scale_factor=0.8)
        self.add(circle)
        
        self.play(
            circle.animate.scale(2.0),
            run_time=2
        )
        self.play(
            circle.animate.scale(0.25),
            run_time=2
        )
