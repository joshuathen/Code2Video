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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the scene
        self.setup_layout(
            "The Macroscopic Mystery", 
            [
                "Light travels slower through glass than in a vacuum.", 
                "We define this using the refractive index, n.", 
                "But how does light actually slow down inside matter?", 
                "Like a lifeguard on sand, light minimizes travel time.", 
                "This path-bending behavior is called Fermat's Principle."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        vacuum_label = Text("Vacuum", font_size=20)
        self.place_at_grid(vacuum_label, "A1", scale_factor=0.8)
        
        glass_label = Text("Glass", font_size=20)
        self.place_at_grid(glass_label, "D1", scale_factor=0.8)
        
        glass_rect = Rectangle(width=5.5, height=2, fill_color=WHITE, fill_opacity=0.3, stroke_width=0)
        self.place_in_area(glass_rect, 'D1', 'F6')
        
        # Particles to represent light beams
        v_dot = Dot(color=WHITE)
        g_dot = Dot(color=WHITE)
        
        # Movement trackers
        v_tracker = ValueTracker(0)
        g_tracker = ValueTracker(0)
        
        v_start = self.grid['B1'] + LEFT * 0.5
        v_end = self.grid['B6'] + RIGHT * 0.5
        g_start = self.grid['E1'] + LEFT * 0.5
        g_end = self.grid['E6'] + RIGHT * 0.5
        
        v_dot.add_updater(lambda d: d.move_to(interpolate(v_start, v_end, v_tracker.get_value())))
        g_dot.add_updater(lambda d: d.move_to(interpolate(g_start, g_end, g_tracker.get_value())))
        
        self.add(vacuum_label, glass_label, glass_rect, v_dot, g_dot)
        
        # Faster in vacuum, slower in glass (n=1.5)
        self.play(
            v_tracker.animate(run_time=2, rate_func=linear).set_value(1),
            g_tracker.animate(run_time=2, rate_func=linear).set_value(0.66),
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Clear previous for formula
        self.play(FadeOut(v_dot, g_dot, vacuum_label, glass_label, glass_rect))
        
        # Formula: n = c / v (Constructed without MathTex to avoid LaTeX dependencies)
        n_part = Text("n", font_size=60, color=WHITE)
        eq_part = Text("=", font_size=60, color=WHITE)
        c_part = Text("c", font_size=60, color=YELLOW)
        v_part = Text("v", font_size=60, color=TEAL)
        div_line = Line(LEFT * 0.4, RIGHT * 0.4, stroke_width=2, color=WHITE)
        fraction = VGroup(c_part, div_line, v_part).arrange(DOWN, buff=0.1)
        formula = VGroup(n_part, eq_part, fraction).arrange(RIGHT, buff=0.2)
        
        self.place_in_area(formula, 'C2', 'D5')
        
        label_c = Text("Speed in Vacuum", font_size=18, color=YELLOW)
        label_v = Text("Speed in Medium", font_size=18, color=TEAL)
        
        self.place_at_grid(label_c, 'B3', scale_factor=1.0)
        self.place_at_grid(label_v, 'E4', scale_factor=1.0)
        
        self.play(Write(formula), Write(label_c), Write(label_v))
        self.wait(2)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(FadeOut(formula, label_c, label_v))
        
        # Magnified Glass with Atoms
        mag_glass = Rectangle(width=5.5, height=4, fill_color=WHITE, fill_opacity=0.1, stroke_width=2)
        self.place_in_area(mag_glass, 'A1', 'F6')
        
        # Use a reproducible seed for random atom positions
        np.random.seed(42)
        atoms = VGroup(*[Dot(radius=0.1, color=WHITE) for _ in range(15)])
        for atom in atoms:
            atom.move_to(
                self.grid['C3'] + np.array([np.random.uniform(-2, 2), np.random.uniform(-1.5, 1.5), 0])
            )
            
        wave_dot = Dot(radius=0, color=WHITE)
        self.place_at_grid(wave_dot, 'C1')
        wave = TracedPath(lambda: wave_dot.get_center(), stroke_color=WHITE, stroke_width=4)
        
        # Oscillating wave motion
        wave_time = ValueTracker(0)
        wave_dot.add_updater(lambda d: d.move_to(
            self.grid['C1'] + RIGHT * wave_time.get_value() * 1.5 + UP * 0.5 * np.sin(wave_time.get_value() * 5)
        ))
        
        self.play(FadeIn(mag_glass), Create(atoms))
        self.add(wave_dot, wave)
        self.play(wave_time.animate(run_time=4, rate_func=linear).set_value(3.5))
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        self.play(FadeOut(mag_glass, atoms, wave, wave_dot))
        
        # Split screen: Lifeguard vs Light Ray
        sand = Rectangle(width=2.5, height=2.5, fill_color="#F4A460", fill_opacity=0.8, stroke_width=0)
        water = Rectangle(width=2.5, height=2.5, fill_color="#0000FF", fill_opacity=0.5, stroke_width=0)
        self.place_in_area(sand, 'A1', 'C3')
        self.place_in_area(water, 'D1', 'F3')
        
        vacuum_side = Rectangle(width=2.5, height=2.5, fill_color=BLACK, stroke_width=0)
        glass_side = Rectangle(width=2.5, height=2.5, fill_color=WHITE, fill_opacity=0.2, stroke_width=0)
        self.place_in_area(vacuum_side, 'A4', 'C6')
        self.place_in_area(glass_side, 'D4', 'F6')
        
        interface_line = Line(self.grid['C1'], self.grid['C6'], color=WHITE)
        
        # Paths
        lg_p1 = self.grid['B1']
        lg_p2 = self.grid['C2']
        lg_p3 = self.grid['E1']
        lg_path = Line(lg_p1, lg_p2, color=WHITE).append_points(Line(lg_p2, lg_p3, color=WHITE).points)
        
        lr_p1 = self.grid['B4']
        lr_p2 = self.grid['C5']
        lr_p3 = self.grid['E4']
        lr_path = Line(lr_p1, lr_p2, color=WHITE).append_points(Line(lr_p2, lg_p3 + RIGHT * 3, color=WHITE).points)
        # Recalculate p3 for light ray to stay within reasonable bounds
        lr_p3_fixed = self.grid['E4']
        lr_path = Line(lr_p1, lr_p2, color=WHITE).append_points(Line(lr_p2, lr_p3_fixed, color=WHITE).points)
        
        self.play(FadeIn(sand, water, vacuum_side, glass_side, interface_line))
        self.play(Create(lg_path), Create(lr_path))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Highlight bent path
        lg_highlight = Line(lg_p1, lg_p2, color=YELLOW).append_points(Line(lg_p2, lg_p3, color=YELLOW).points)
        lr_highlight = Line(lr_p1, lr_p2, color=YELLOW).append_points(Line(lr_p2, lr_p3_fixed, color=YELLOW).points)
        
        # Dashed straight line comparison
        dashed_straight = DashedLine(lr_p1, self.grid['E6'], color=GRAY)
        
        fermat_text = Text("Fermat's Principle: Least Time", font_size=24, color=WHITE)
        self.place_in_area(fermat_text, 'F1', 'F6')
        
        self.play(
            Create(lg_highlight), 
            Create(lr_highlight), 
            Create(dashed_straight),
            Write(fermat_text)
        )
        self.wait(2)
