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
            "Elastic collisions conserve both momentum and energy.", 
            "Conservation laws dictate the resulting velocity.", 
            "Velocity vectors show energy transfer between blocks."
        ])
        
        # Load assets
        block_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=BLUE)
        block_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=BLUE)
        label_m = Text("m", color=WHITE, font_size=20).next_to(block_a, UP)
        label_M = Text("M", color=WHITE, font_size=20).next_to(block_b, UP)
        blocks = VGroup(block_a, label_m, block_b, label_M)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(block_a, 'D3', scale_factor=1.0)
        self.place_at_grid(block_b, 'D4', scale_factor=1.0)
        self.play(FadeIn(blocks))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED)
        velocity_vector = Arrow(start=LEFT*0.5, end=RIGHT*0.5, color=YELLOW)
        self.place_at_grid(velocity_vector, 'D3', scale_factor=0.8)
        
        # Collision flash
        flash = Dot(self.grid['D4'], color=RED, radius=0.5)
        self.play(Create(velocity_vector), Flash(flash, color=RED))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        eq = MathTex(r"K = \frac{1}{2}mv^2", color=WHITE)
        self.place_at_grid(eq, 'B3', scale_factor=1.2)
        self.play(Write(eq))
        self.wait(2)
