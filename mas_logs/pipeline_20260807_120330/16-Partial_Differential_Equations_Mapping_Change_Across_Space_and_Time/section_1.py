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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the scene
        title = "Prerequisite Bridge: From ODEs to PDEs"
        lines = [
            "ODEs track changes over a single variable, like time.",
            "Think of an ant moving along a thin wire.",
            "PDEs describe systems changing across space and time simultaneously.",
            "Imagine ink spreading through a pool of water.",
            "Here, concentration depends on both position and time."
        ]
        self.setup_layout(title, lines)

        # Colors
        color_ode = "#FFFF00"  # Yellow
        color_pde = "#87CEEB"  # Light Blue
        color_pool = "#0000FF" # Blue
        color_ink = "#A9A9A9"  # Dark Gray
        color_label = "#00FF00" # Green

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_ode))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_ode))
        
        # 1D Wire (Stretches across columns to show path)
        wire = Line(self.grid["C1"], self.grid["C6"], color=GREY)
        wire_label = MathTex("x(t)", color=color_ode)
        # Issue 30: Place wire_label at B3
        self.place_at_grid(wire_label, "B3", scale_factor=1.0)
        
        # Ant Asset
        # Issue 25: Load ant asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg]
        ant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg", color=color_ode)
        # Issue 30: Place ant at C3
        self.place_at_grid(ant, "C3", scale_factor=0.4)
        
        self.play(Create(wire), FadeIn(wire_label))
        self.play(FadeIn(ant))
        # Move ant along the wire to C6
        self.play(ant.animate.move_to(self.grid["C6"]), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[2].animate.set_color(color_pde),
            FadeOut(wire), FadeOut(wire_label), FadeOut(ant)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(color_pool))
        
        # Pool Asset
        # Issue 25: Load pool asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pool.svg]
        # Issue 31: Place pool in area B2 to E5
        pool = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pool.svg", color=color_pool)
        self.place_in_area(pool, "B2", "E5", scale_factor=2.0)
        
        # Ink (Spreading Dot/Circle) - Persistent mobject with ValueTracker
        ink_center = pool.get_center()
        radius_tracker = ValueTracker(0.1)
        ink = Dot(point=ink_center, radius=0.1, color=color_ink, fill_opacity=0.7)
        ink.add_updater(lambda m: m.set_width(radius_tracker.get_value() * 2) if radius_tracker.get_value() > 0 else m)
        
        self.play(FadeIn(pool))
        self.add(ink)
        self.play(radius_tracker.animate.set_value(1.5), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(color_label))
        
        # Concentration Label
        # Issue 32: Place formula at A3, scale 1.2
        formula = MathTex("u(x, y, t)", color=color_label)
        self.place_at_grid(formula, "A3", scale_factor=1.2)
        
        self.play(Write(formula))
        self.wait(3)
