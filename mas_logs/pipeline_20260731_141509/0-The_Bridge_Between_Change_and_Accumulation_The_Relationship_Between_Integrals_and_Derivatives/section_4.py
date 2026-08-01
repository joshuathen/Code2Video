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

class Section4Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "The Fundamental Theorem: The Mathematical U-Turn"
        lines = [
            "The Fundamental Theorem links differentiation and integration.",
            "They are perfect inverse operations of each other.",
            "Integrating a rate gives the net total change.",
            "Differentiating an accumulation yields the original rate.",
            "It’s a mathematical U-turn connecting speed and distance."
        ]
        self.setup_layout(title, lines)

        # Helper to create a gear
        def create_gear(color, radius=0.3):
            core = Circle(radius=radius, color=color, fill_opacity=0.3, stroke_width=2)
            teeth_count = 12
            teeth = VGroup()
            for i in range(teeth_count):
                angle = i * (2 * PI / teeth_count)
                tooth = Rectangle(width=0.1, height=0.15, color=color, fill_opacity=0.8, stroke_width=0)
                tooth.rotate(angle)
                tooth.move_to(core.get_center() + radius * np.array([np.cos(angle), np.sin(angle), 0]))
                teeth.add(tooth)
            return VGroup(core, teeth)

        # === Animation for Lecture Line 1 ===
        # A 'Math Machine' icon (#A9A9A9) with two connected gears appears.
        self.lecture[0].set_color(YELLOW)
        
        # Asset integration (Issue 27) and repositioning (Issue 44)
        machine_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg", color="#A9A9A9")
        self.place_in_area(machine_asset, "A3", "E6", scale_factor=1.8)
        
        gear1 = create_gear("#A9A9A9", radius=0.3)
        gear2 = create_gear("#A9A9A9", radius=0.3)
        
        self.place_at_grid(gear1, "B4")
        self.place_at_grid(gear2, "D4")
        
        # Rotate gears continuously
        gear1.add_updater(lambda m, dt: m.rotate(dt * 1.5))
        gear2.add_updater(lambda m, dt: m.rotate(-dt * 1.5))
        
        machine_label = Text("MATH MACHINE", font_size=18, color="#A9A9A9")
        self.place_at_grid(machine_label, "A4")
        
        machine_group = VGroup(machine_asset, gear1, gear2, machine_label)
        self.play(FadeIn(machine_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # An arrow labeled 'f(x)' enters Gear 1, outputting 'f'(x)' (#00FF00).
        self.lecture[1].set_color(BLUE)
        
        f_x_in = Text("f(x)", font_size=24, color=WHITE)
        self.place_at_grid(f_x_in, "B3") # Issue 45
        
        arrow_in = Arrow(start=self.grid["B3"], end=self.grid["B4"], color=WHITE, buff=0.4)
        
        f_prime_out = Text("f'(x)", font_size=24, color="#00FF00")
        self.place_at_grid(f_prime_out, "B5") # Issue 45
        
        arrow_out1 = Arrow(start=self.grid["B4"], end=self.grid["B5"], color="#00FF00", buff=0.4)
        
        self.play(
            FadeIn(f_x_in),
            Create(arrow_in)
        )
        self.play(
            Create(arrow_out1),
            FadeIn(f_prime_out)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # 'f'(x)' enters Gear 2, outputting 'f(x)' (#FF4500) again.
        self.lecture[2].set_color(GREEN)
        
        # Path from f'(x) at B5 down to Gear 2 path at D5 then into Gear 2 at D4
        connector_down = Line(self.grid["B5"], self.grid["D5"], color="#00FF00")
        arrow_into_gear2 = Arrow(start=self.grid["D5"], end=self.grid["D4"], color="#00FF00", buff=0.4)
        
        f_x_final = Text("f(x)", font_size=24, color="#FF4500")
        self.place_at_grid(f_x_final, "D3") # Issue 45
        
        arrow_out2 = Arrow(start=self.grid["D4"], end=self.grid["D3"], color="#FF4500", buff=0.4)
        
        self.play(Create(connector_down))
        self.play(Create(arrow_into_gear2))
        self.play(
            Create(arrow_out2),
            FadeIn(f_x_final)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The formula ∫[a,b] f'(x)dx = f(b) - f(a) appears in gold (#D4AF37).
        self.lecture[3].set_color(ORANGE)
        
        # Using Text for formula as a fallback for potential LaTeX issues as per B008
        # but the prompt asked for high quality, so I'll try MathTex first.
        # Given "status: ok" in render context, MathTex should work.
        ftc_formula = MathTex(r"\int_a^b f'(x)dx = f(b) - f(a)", color="#D4AF37")
        self.place_in_area(ftc_formula, "F3", "F6", scale_factor=0.8) # Issue 46
        
        self.play(Write(ftc_formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The 'U-Turn' arrow (#FFFFFF) connects integration back to differentiation.
        self.lecture[4].set_color(WHITE)
        
        # U-turn from f(x) final (D3) back to f(x) in (B3)
        u_turn_arrow = CurvedArrow(
            start_point=self.grid["D3"],
            end_point=self.grid["B3"],
            color="#FFFFFF",
            angle=-PI/2
        )
        u_turn_label = Text("U-TURN", font_size=16, color=WHITE)
        self.place_at_grid(u_turn_label, "C3", scale_factor=0.6) # Issue 46
        
        self.play(
            Create(u_turn_arrow),
            FadeIn(u_turn_label)
        )
        self.wait(2)
