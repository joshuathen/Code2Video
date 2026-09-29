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
        self.setup_layout("Application: Thermal Diffusion in a Ring", [
            "Heat flows naturally within a ring.",
            "Boundary conditions simplify circular thermal flow.",
            "Fourier series describe steady temperature states."
        ])
        
        # Load assets
        wire = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wire.svg")
        ring = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ring.svg")
        
        ring_label = Text("Ring", font_size=24, color=WHITE)
        bc_text = MathTex("u(0, t) = u(L, t)", color="#2ECC71")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#E74C3C"))
        
        # Placing and showing ring elements
        self.place_at_grid(ring, 'C3', scale_factor=1.2)
        self.place_at_grid(ring_label, 'B3', scale_factor=1.0)
        self.add(ring, ring_label)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#3498DB"))
        
        # Illustrate wire/diffusion
        wire_visual = self.place_at_grid(wire, 'D3', scale_factor=0.5)
        arrows = VGroup(*[Arrow(start=UP, end=RIGHT, color="#3498DB").scale(0.3) for _ in range(4)])
        arrows.arrange_in_grid(2, 2)
        self.place_at_grid(arrows, 'D4', scale_factor=0.8)
        self.play(Create(wire_visual), Create(arrows))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#2ECC71"))
        
        # Display periodic boundary condition
        self.place_in_area(bc_text, 'E2', 'F4', scale_factor=0.9)
        self.play(Write(bc_text))
        
        self.wait(2)
