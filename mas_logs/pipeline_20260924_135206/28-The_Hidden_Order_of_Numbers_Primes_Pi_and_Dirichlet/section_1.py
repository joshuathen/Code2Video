from manim import *
import numpy as np

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
        lecture_lines = [
            "Primes are the fundamental atoms of arithmetic.",
            "They appear random, yet follow asymptotic patterns.",
            "Sieve of Eratosthenes filters integers.",
            "Primes become sparser as numbers grow.",
            "Order emerges from apparent chaotic distribution."
        ]
        self.setup_layout("The Elusive Prime Patterns", lecture_lines)
        
        # Assets
        atom_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/atom.svg"
        hist_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/histogram.svg"

        # === Animation for Lecture Line 1 ===
        num_line = VGroup(*[Text(str(i), font_size=16) for i in range(1, 21)])
        num_line.arrange(RIGHT, buff=0.2)
        self.place_in_area(num_line, 'C1', 'C6', scale_factor=0.8)
        
        atom = SVGMobject(atom_path)
        self.place_at_grid(atom, 'B1', scale_factor=0.3)
        atom.set_color(WHITE)
        
        primes = [2, 3, 5, 7]
        for i in primes:
            num_line[i-1].set_color("#FFD700")
            
        self.play(self.lecture[0].animate.set_color("#FFD700"), run_time=1)
        self.play(FadeIn(atom), FadeIn(num_line))

        # === Animation for Lecture Line 2 ===
        hist = SVGMobject(hist_path)
        self.place_at_grid(hist, 'B4', scale_factor=0.3)
        self.play(self.lecture[1].animate.set_color("#87CEEB"), run_time=1)
        self.play(FadeIn(hist))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        sieve_grid = VGroup(*[Square(side_length=0.4).set_fill(BLUE, opacity=0.3) for _ in range(25)])
        sieve_grid.arrange_in_grid(rows=5, cols=5, buff=0.1)
        self.place_at_grid(sieve_grid, 'D4', scale_factor=0.7)
        self.play(self.lecture[2].animate.set_color("#FFD700"), run_time=1)
        self.play(FadeIn(sieve_grid))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        axes = Axes(x_range=[0, 10, 2], y_range=[0, 1, 0.5], axis_config={"include_tip": False})
        self.place_in_area(axes, 'E1', 'F3', scale_factor=0.5)
        graph = axes.plot(lambda x: 1/np.sqrt(x+1), color=WHITE)
        self.play(self.lecture[3].animate.set_color("#FF6347"), run_time=1)
        self.play(Create(axes), Create(graph))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        atom_2 = SVGMobject(atom_path)
        self.place_in_area(atom_2, 'E4', 'F6', scale_factor=0.5)
        atom_2.set_color("#FFD700")
        
        self.play(self.lecture[4].animate.set_color("#00FF7F"), run_time=1)
        self.play(FadeIn(atom_2))
        self.wait(2)
