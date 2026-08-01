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
        self.setup_layout(
            "The Feedback Loop: Iteration", 
            [
                "Iteration feeds a function's output back as its next input.", 
                "The sequence of positions creates a path called an orbit.", 
                "These orbits reveal the long-term behavior of the system."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        formula = Text("z_{n+1} = f(z_n)", color="#FFFFFF")
        # Place formula at the top-center of the grid area
        # Issue 30: Scale factor reduced from 1.2 to 1.0 to avoid crowding
        self.place_in_area(formula, "A1", "A6", scale_factor=1.0)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Define orbit points using the grid
        p0_pos = self.grid["D2"]
        p1_pos = self.grid["B3"]
        p2_pos = self.grid["C5"]
        p3_pos = self.grid["E4"]
        
        # Create persistent dot
        dot = Dot(color=WHITE, radius=0.1)
        self.place_at_grid(dot, "D2")
        
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        z0_label = Text("z_0", font_size=28, color=WHITE)
        self.place_at_grid(z0_label, "E2", scale_factor=1.0)
        
        self.play(FadeIn(dot), FadeIn(z0_label))
        
        # Magenta transition from z0 to z1
        arrow1 = CurvedArrow(p0_pos, p1_pos, color="#FF00FF", angle=-TAU/8)
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        z1_label = Text("z_1", font_size=28, color=WHITE)
        # Issue 29: Moved from A3 to B1 to avoid overlap with formula
        self.place_at_grid(z1_label, "B1", scale_factor=1.0)
        
        self.play(Create(arrow1))
        self.play(
            MoveAlongPath(dot, arrow1),
            FadeIn(z1_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Start trailing
        trail = TracedPath(dot.get_center, stroke_color="#00FF00", stroke_width=4)
        self.add(trail)
        
        # Transition to z2
        arrow2 = CurvedArrow(p1_pos, p2_pos, color="#00FF00", angle=-TAU/8)
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        z2_label = Text("z_2", font_size=28, color=WHITE)
        # Issue 31: Moved from B5 to C5 to avoid clutter near formula area
        self.place_at_grid(z2_label, "C5", scale_factor=1.0)
        
        self.play(Create(arrow2))
        self.play(
            MoveAlongPath(dot, arrow2),
            FadeIn(z2_label),
            run_time=1.2
        )
        
        # Transition to z3
        arrow3 = CurvedArrow(p2_pos, p3_pos, color="#00FF00", angle=-TAU/8)
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        z3_label = Text("z_3", font_size=28, color=WHITE)
        self.place_at_grid(z3_label, "F4", scale_factor=1.0)
        
        self.play(Create(arrow3))
        self.play(
            MoveAlongPath(dot, arrow3),
            FadeIn(z3_label),
            run_time=1.2
        )
        
        self.wait(3)
