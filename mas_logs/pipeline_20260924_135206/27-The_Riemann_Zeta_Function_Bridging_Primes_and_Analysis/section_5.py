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
        lecture_lines = ["Zeros dictate the distribution of prime numbers.", "Proving the hypothesis reveals prime number structure.", "We predict primes like seeds in a garden."]
        self.setup_layout("Why It Matters: The Music of the Primes", lecture_lines)
        
        # Assets
        seeds = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/seeds.svg")
        garden = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/garden.svg")
        
        # Elements
        primes = VGroup(*[Circle(radius=0.2, color=color, fill_opacity=0.8) for color in ["#FF0000", "#00FF00", "#0000FF", "#FF0000", "#00FF00"]])
        primes.arrange(RIGHT, buff=0.2)
        primes_with_seeds = VGroup(primes, seeds)
        self.place_in_area(primes_with_seeds, 'B4', 'C6', scale_factor=0.4)
        
        zeros = VGroup(*[Dot(color="#ADD8E6") for _ in range(5)])
        zeros.arrange(RIGHT, buff=0.5)
        self.place_at_grid(zeros, 'B2', scale_factor=1.2)
        
        wave = FunctionGraph(lambda x: 0.5 * np.sin(3 * x), x_range=[-2, 2], color="#FFD700")
        garden_with_wave = VGroup(garden, wave)
        self.place_at_grid(garden_with_wave, 'E5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(primes_with_seeds))
        self.lecture[0].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(zeros), FadeTransform(primes.copy(), zeros))
        self.lecture[1].set_color("#ADD8E6")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Create(garden_with_wave))
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
