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
        self.setup_layout("The Bridge: Primes and Zeros", [
            "Euler links Zeta to the prime numbers.",
            "Primes behave like a chaotic swarm.",
            "Zeta zeros reveal the underlying prime rhythm."
        ])
        
        # Mobjects
        primes = VGroup(*[Dot(color=BLUE) for _ in range(10)]).arrange_in_grid(2, 5, buff=0.2)
        zeros = VGroup(*[Dot(color=RED) for _ in range(5)]).arrange(DOWN, buff=0.4)
        swarm = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/swarm.svg")
        
        # Positions adjusted per feedback
        self.place_at_grid(primes, 'D2', scale_factor=0.9)
        self.place_at_grid(zeros, 'D5', scale_factor=0.9)
        
        # Connection
        connection = DashedLine(primes.get_right(), zeros.get_left(), color=ORANGE)
        self.place_in_area(connection, 'C2', 'C5', scale_factor=0.9)
        self.place_at_grid(swarm, 'C3', scale_factor=0.3)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(primes))
        self.play(FadeIn(swarm))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(zeros))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        self.play(Create(connection))
        self.play(Indicate(connection, color=RED))
        self.wait(2)
