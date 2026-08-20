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
        lecture_lines = [
            "Mass ratios define Pi's digits.",
            "One hundred powers yield Pi.",
            "Discrete hits reveal transcendental constants."
        ]
        self.setup_layout("The Connection: Why Pi?", lecture_lines)
        
        # Color palettes for highlighting
        colors = [BLUE, GREEN, YELLOW]
        
        # Assets
        block_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        collision_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/collision.svg")

        # Create visual elements
        table = Table(
            [["Ratio (M/m)", "Collisions"], ["1", "3"], ["100", "31"], ["10,000", "314"]],
            include_outer_lines=True
        )
        
        pi_digits = MathTex(r"\pi \approx 3.14159...").set_color(ORANGE)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        # Show mass ratio with block icon
        self.place_at_grid(block_icon, 'A4', scale_factor=0.5)
        self.play(FadeIn(block_icon))
        
        # Show table in area B4-C5
        self.place_in_area(table, 'B4', 'C5', scale_factor=0.65)
        self.play(Create(table))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        # Pi digits with collision icon
        self.place_at_grid(collision_icon, 'E2', scale_factor=0.5)
        self.place_at_grid(pi_digits, 'E4', scale_factor=0.7)
        self.play(FadeIn(collision_icon), Write(pi_digits))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        # Flash relationship
        self.play(Indicate(table), Indicate(pi_digits), Indicate(collision_icon))
        self.wait(2)
