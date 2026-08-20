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
        self.setup_layout("The Linear System as a Geometric Puzzle", [
            "Systems of equations form a geometric puzzle.",
            "Represent Ax equals b as a vector combination.",
            "We solve for weights reaching the target vector."
        ])

        # Define vectors
        axes = Axes(x_range=[-1, 6], y_range=[-1, 6], axis_config={"include_tip": True}).scale(0.5)
        # Issue 23/36 Fix: Position axes
        self.place_at_grid(axes, 'B3', scale_factor=0.7)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/puzzle.svg
        puzzle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/puzzle.svg")
        self.place_at_grid(puzzle_icon, 'A5', scale_factor=0.5)
        
        v1 = Vector([2, 1], color=YELLOW)
        v2 = Vector([1, 2], color=BLUE)
        b = Vector([5, 5], color=RED)
        
        # Manually align vectors to the coordinate system
        v1.move_to(axes.c2p(0,0)).shift(axes.c2p(2,1)-axes.c2p(0,0))
        v2.move_to(axes.c2p(0,0)).shift(axes.c2p(1,2)-axes.c2p(0,0))
        b.move_to(axes.c2p(0,0)).shift(axes.c2p(5,5)-axes.c2p(0,0))
        
        v1_label = MathTex(r"\\vec{v}_1", color=YELLOW)
        v2_label = MathTex(r"\\vec{v}_2", color=BLUE)
        b_label = MathTex(r"\\vec{b}", color=RED)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(axes), FadeIn(puzzle_icon))
        self.play(Create(v1), Create(v2), Create(b))
        
        # Issue 25/36 Fix: Position labels
        self.place_at_grid(v1_label, 'C4', scale_factor=0.6)
        self.place_at_grid(v2_label, 'B4', scale_factor=0.6)
        self.add(b_label.next_to(b.get_end(), UP))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        # Visually demonstrate linear combination x1*v1 + x2*v2 = b
        v1_scaled = Vector([1.66 * 2, 1.66 * 1], color=YELLOW).move_to(axes.c2p(1.66, 0.83))
        v2_scaled = Vector([1.66 * 1, 1.66 * 2], color=BLUE).move_to(axes.c2p(0.83, 1.66))
        self.play(v1.animate.become(v1_scaled), v2.animate.become(v2_scaled))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED)
        # Flash the weights
        weight_text = Text("x1=1.66, x2=1.66", color=WHITE, font_size=20)
        # Issue 24/36 Fix: Position weight_text
        self.place_at_grid(weight_text, 'D3', scale_factor=0.9)
        self.play(Flash(weight_text))
        self.wait(1)
