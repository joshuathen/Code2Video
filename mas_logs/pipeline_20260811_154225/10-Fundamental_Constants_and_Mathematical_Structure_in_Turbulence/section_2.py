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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Large eddies break down into smaller ones.",
            "Kinetic energy cascades downward until viscous dissipation.",
            "The inertial range is scale-invariant physics."
        ]
        self.setup_layout("The Kolmogorov Cascade Hypothesis", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/eddy.svg]
        large_eddy = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eddy.svg", color="#00FFFF")
        label1 = Text("Large Eddy", font_size=24 * 0.7, color="#00FFFF")
        label1.next_to(large_eddy, DOWN)
        group1 = VGroup(large_eddy, label1)
        self.place_at_grid(group1, 'B2', scale_factor=0.6)
        self.play(Create(group1))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        smaller_eddies = VGroup(*[
            Circle(radius=0.5 - i*0.1, color="#FFFF00", stroke_width=3) 
            for i in range(3)
        ]).arrange(RIGHT, buff=0.2)
        label2 = Text("Energy Cascade", font_size=24 * 0.7, color="#FFFF00")
        label2.next_to(smaller_eddies, DOWN)
        group2 = VGroup(smaller_eddies, label2)
        self.place_at_grid(group2, 'B4', scale_factor=0.5)
        self.play(Create(group2))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/eddy.svg]
        tiny_eddies = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eddy.svg", color="#FF4500")
        dots = VGroup(*[tiny_eddies.copy().scale(0.1) for _ in range(10)])
        dots.arrange_in_grid(2, 5, buff=0.1)
        self.place_in_area(dots, 'D4', 'F6', scale_factor=0.8)
        self.play(FadeIn(dots))
        self.lecture[2].set_color("#FF4500")
        self.wait(2)
