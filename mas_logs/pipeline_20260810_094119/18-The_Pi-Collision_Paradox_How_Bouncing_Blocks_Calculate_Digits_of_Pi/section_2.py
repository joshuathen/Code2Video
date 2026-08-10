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
        self.setup_layout("Prerequisite Physics: Conservation Laws", [
            "Collisions follow conservation of momentum.",
            "Kinetic energy is also conserved.",
            "These laws dictate velocity exchanges."
        ])
        
        # Load Assets
        pendulum_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg")
        
        # Text Objects
        p1 = Text("P_1", color=WHITE, font_size=24)
        p2 = Text("P_2", color=WHITE, font_size=24)
        ke1 = Text("KE_1", color="#00FFFF", font_size=24)
        ke2 = Text("KE_2", color="#00FFFF", font_size=24)
        
        momentum_group = VGroup(p1, p2, pendulum_icon)
        
        # Position using area layout for better spacing
        self.place_in_area(momentum_group, 'B2', 'D5', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        self.play(FadeIn(momentum_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(FadeIn(ke1), FadeIn(ke2))
        self.place_at_grid(ke1, 'E2', scale_factor=0.8)
        self.place_at_grid(ke2, 'E5', scale_factor=0.8)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        arrow1 = Arrow(start=pendulum_icon.get_center(), end=ke1.get_center(), color="#FFFF00")
        arrow2 = Arrow(start=pendulum_icon.get_center(), end=ke2.get_center(), color="#FFFF00")
        
        self.play(Create(arrow1), Create(arrow2))
        self.wait(2)
