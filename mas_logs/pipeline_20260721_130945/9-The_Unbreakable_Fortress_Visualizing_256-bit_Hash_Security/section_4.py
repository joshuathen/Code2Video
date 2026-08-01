from manim import *
import random

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
        title_text = "The Brute Force Challenge: Time vs. Computation"
        lecture_lines = [
            "Brute force means trying every possible combination.",
            "Even a billion supercomputers would take trillions of years.",
            "This exceeds the total lifespan of our entire universe."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_ROBOT = "#C0C0C0"
        COLOR_TIMER = "#FF0000"
        
        # === Animation for Lecture Line 1 ===
        # A 'Super-Robot' icon (#C0C0C0) rapidly flashes through random hex strings.
        self.play(self.lecture[0].animate.set_color(COLOR_ROBOT))
        
        # Create Robot Icon using Asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").set_color(COLOR_ROBOT)
        self.place_at_grid(robot, "C3", scale_factor=1.0)
        
        self.play(FadeIn(robot))
        
        # Hex strings - Scale 0.5 to avoid overlap (Issue 34)
        hex_strings = [
            Text("0x" + "".join(random.choices("0123456789ABCDEF", k=8)), 
                 font_size=20, color=COLOR_ROBOT)
            for _ in range(15)
        ]
        for hs in hex_strings:
            self.place_at_grid(hs, "C4", scale_factor=0.5)
        
        # Rapid flashing effect
        for hs in hex_strings:
            self.add(hs)
            self.wait(0.1)
            self.remove(hs)
        
        # === Animation for Lecture Line 2 ===
        # Multiply robot into a grid; a red timer (#FF0000) counts 'Trillions of Years'.
        self.play(self.lecture[1].animate.set_color(COLOR_TIMER))
        
        # Robot Grid - Expand area and reduce scale to avoid crowding (Issue 33)
        robots_grid = VGroup(*[robot.copy() for _ in range(12)])
        robots_grid.arrange_in_grid(rows=3, cols=4, buff=0.3)
        self.place_in_area(robots_grid, 'B2', 'F6', scale_factor=0.6)
        
        # Timer Group - Wide area and small scale to prevent collision (Issue 32)
        timer_label = Text("Time Elapsed:", font_size=24, color=COLOR_TIMER)
        timer_value = DecimalNumber(0, num_decimal_places=0, color=COLOR_TIMER, include_sign=False)
        timer_unit = Text("Trillion Years", font_size=20, color=COLOR_TIMER)
        timer_group = VGroup(timer_label, timer_value, timer_unit).arrange(RIGHT, buff=0.2)
        
        self.place_in_area(timer_group, 'A1', 'A6', scale_factor=0.6)
        
        self.play(
            FadeOut(robot),
            FadeIn(robots_grid),
            FadeIn(timer_group)
        )
        
        # Counter animation
        self.play(
            ChangeDecimalToValue(timer_value, 1000),
            run_time=2,
            rate_func=linear
        )
        
        # === Animation for Lecture Line 3 ===
        # Background stars fade to black as the timer continues to spin.
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Stars Mobjects - Background element
        stars = VGroup(*[Dot(radius=0.01, color=WHITE, fill_opacity=random.uniform(0.3, 0.8)) 
                         for _ in range(60)])
        for star in stars:
            # Random positioning within right-side grid area
            star.move_to([random.uniform(0.5, 6.0), random.uniform(-2.5, 2.0), 0])
            
        self.play(FadeIn(stars))
        
        # Final sequence: stars fade out, timer explodes in value
        self.play(
            stars.animate.set_fill(opacity=0),
            ChangeDecimalToValue(timer_value, 10**12),
            run_time=4,
            rate_func=linear
        )
        
        self.wait(2)
