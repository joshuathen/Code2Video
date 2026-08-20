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
        self.setup_layout("Application: The Cosmic Clock", [
            "64 disks take forever to move.",
            "Exponential growth outpaces linear growth.",
            "Complexity scales with binary counting."
        ])
        
        # Asset path
        clock_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg"
        
        # === Animation for Lecture Line 1 ===
        # Draw circular clock using SVG
        clock = SVGMobject(clock_path, color=BLUE)
        self.place_at_grid(clock, "D4", scale_factor=0.6)
        
        # Creating a hand for the SVG clock
        clock_hand = Line(ORIGIN, UP * 0.8, color=YELLOW)
        clock_hand.move_to(clock.get_center(), aligned_edge=DOWN)
        
        self.play(FadeIn(clock), GrowFromCenter(clock_hand))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Exponential growth: Hare vs Turtle
        hare = Dot(color=RED, radius=0.15)
        turtle = Dot(color=GREEN, radius=0.15)
        self.place_at_grid(hare, "F2", scale_factor=0.5)
        self.place_at_grid(turtle, "F5", scale_factor=0.5)
        
        self.play(FadeIn(hare), FadeIn(turtle))
        self.play(
            hare.animate.shift(RIGHT * 2),
            turtle.animate.shift(RIGHT * 1),
            run_time=2
        )
        self.lecture[1].set_color(RED)

        # === Animation for Lecture Line 3 ===
        # Clock hand rotating representing binary complexity
        self.play(Rotate(clock_hand, angle=PI, about_point=clock.get_center()), run_time=2)
        self.lecture[2].set_color(YELLOW)
        self.wait(1)
