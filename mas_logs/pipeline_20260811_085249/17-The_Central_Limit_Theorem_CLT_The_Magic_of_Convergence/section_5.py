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
        lecture_lines = [
            "The CLT enables powerful statistical hypothesis testing.",
            "Confidence intervals become reliable for unknown populations.",
            "We can predict behavior from irregular data sets."
        ]
        self.setup_layout("Real-World Application: The Law of Averages", lecture_lines)
        
        grid_title = Text("Simulation Grid", font_size=24)
        self.place_in_area(grid_title, 'A3', 'A4', scale_factor=0.8)
        self.add(grid_title)
        
        # === Animation for Lecture Line 1 ===
        # Show a large set of random events using SVGs
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg]
        particle_group = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg") for _ in range(20)])
        self.place_in_area(particle_group, 'D3', 'F6', scale_factor=0.6)
        
        self.play(FadeIn(particle_group))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        # Visualize the average of these events stabilizing
        self.play(particle_group.animate.arrange(buff=0.1).scale(0.5))
        self.lecture[1].set_color("#33FF57")

        # === Animation for Lecture Line 3 ===
        # Label the stable result 'Convergence' in #33FF57 using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg]
        convergence_label = Text("Convergence", color="#33FF57", font_size=32)
        self.place_at_grid(convergence_label, 'C3', scale_factor=0.9)
        
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        coin.scale(0.5).next_to(convergence_label, UP)
        
        self.play(Write(convergence_label), FadeIn(coin))
        self.lecture[2].set_color("#3357FF")
        self.wait(2)
