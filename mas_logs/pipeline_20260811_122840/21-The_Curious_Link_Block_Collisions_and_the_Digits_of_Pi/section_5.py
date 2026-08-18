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
        self.setup_layout("Synthesis & Summary", [
            "Mechanical systems produce transcendental constants.",
            "Conservation laws create geometric shadows of Pi.",
            "Discrete impacts mirror continuous mathematical truth."
        ])
        
        gear = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg")
        pendulum = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700") # Gold
        rect = SurroundingRectangle(self.lecture[0], buff=0.1, color="#FFD700")
        self.play(Create(rect))
        self.place_at_grid(gear, 'A3', scale_factor=0.5)
        self.play(FadeIn(gear), gear.animate.rotate(2*PI), run_time=2)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF") # Cyan
        self.play(FadeOut(rect))
        shadow = Circle(radius=1.5, color="#00FFFF", stroke_width=4)
        self.place_at_grid(shadow, 'B5', scale_factor=0.6)
        self.play(Create(shadow))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF69B4") # Hot Pink
        blocks = VGroup(
            Square(side_length=0.5, color="#FF69B4", fill_opacity=0.5),
            Square(side_length=1.0, color="#FF69B4", fill_opacity=0.5)
        ).arrange(RIGHT, buff=0.2)
        self.place_in_area(blocks, 'D2', 'E5', scale_factor=0.75)
        
        system = VGroup(gear, shadow, blocks, pendulum).set_color(WHITE)
        self.place_at_grid(pendulum, 'F3', scale_factor=0.5)
        self.play(FadeIn(blocks), FadeIn(pendulum), FadeIn(system))
        self.wait(2)
