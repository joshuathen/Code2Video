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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Measures create rhythmic hierarchy.",
            "Beat one is the downbeat.",
            "It acts as the accent.",
            "Higher bounces signify the downbeat.",
            "Marching keeps the squad synchronized."
        ]
        self.setup_layout("Dynamic Application: The Accent Pattern", lecture_lines)
        
        # Assets
        metronome = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg", color=WHITE)
        drumstick = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drumstick.svg", color=WHITE)
        self.place_at_grid(metronome, 'A6', scale_factor=0.5)
        self.add(metronome)

        # Bouncing balls
        balls = VGroup(*[Circle(radius=0.4, color=BLUE, fill_opacity=0.6) for _ in range(4)])
        
        # Layout according to critic instructions
        self.place_in_area(balls[0], 'B3', 'B6', scale_factor=0.8)
        self.place_in_area(balls[1], 'C3', 'C6', scale_factor=0.8)
        self.place_in_area(balls[2], 'D3', 'D6', scale_factor=0.8)
        self.place_in_area(balls[3], 'E3', 'E6', scale_factor=0.8)
        self.add(balls)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_at_grid(drumstick, 'B2', scale_factor=0.5)
        self.add(drumstick)
        self.play(balls[0].animate.set_color("#FF00FF").scale(1.2), drumstick.animate.rotate(-PI/4, about_point=self.grid['B2']), run_time=0.5)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(PURPLE))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(RED))
        self.play(Indicate(balls[0], scale_factor=1.5, color="#FF00FF"))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(TEAL))
        for ball in balls:
            self.play(ball.animate.shift(UP*0.5), run_time=0.2)
            self.play(ball.animate.shift(DOWN*0.5), run_time=0.2)
        self.wait(1)
