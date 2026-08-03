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
        # TEACHING CONTENT
        lecture_lines = [
            "The spheres' contact points on the plane are the foci.",
            "Let's pick any point P on the ellipse's edge.",
            "We will track the distance from P to each focus."
        ]
        self.setup_layout("Defining the Foci", lecture_lines)

        # Colors
        COLOR_FOCI = "#FFA500"
        COLOR_P = "#FF4500"
        COLOR_GEN = "#FFFFFF"

        # Parameters
        # Grid positions for layout (Resolves Issue 35, 36, 37)
        f1_pos = self.grid["D2"]
        f2_pos = self.grid["D5"]
        center_pos = (self.grid["D3"] + self.grid["D4"]) / 2
        
        a = 2.0
        c = 1.5
        b = np.sqrt(a**2 - c**2)
        
        # Vertex V positioned lower (Resolves Issue 35)
        v_pos = (self.grid["B3"] + self.grid["B4"]) / 2 
        
        # Dandelin constants
        vt1 = 0.4
        vt2 = vt1 + 2*a # 4.4
        
        # === Animation for Lecture Line 1 ===
        # Line 1: "The spheres' contact points on the plane are the foci."
        self.lecture[0].set_color(COLOR_FOCI)
        
        # Draw the elliptical intersection
        ellipse = Ellipse(width=2*a, height=2*b, color=WHITE, stroke_width=2).move_to(center_pos)
        
        # Contact points F1 and F2
        f1 = Dot(f1_pos, color=COLOR_FOCI)
        f2 = Dot(f2_pos, color=COLOR_FOCI)
        
        # Grid-aligned labels (Resolves Issue 37)
        f1_label = Text("F1", font_size=20, color=COLOR_FOCI)
        self.place_at_grid(f1_label, "E2", scale_factor=0.8)
        f2_label = Text("F2", font_size=20, color=COLOR_FOCI)
        self.place_at_grid(f2_label, "E5", scale_factor=0.8)
        
        self.play(Create(ellipse))
        self.play(FadeIn(f1, f2, f1_label, f2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: "Let's pick any point P on the ellipse's edge."
        self.lecture[1].set_color(COLOR_P)
        
        # Use Asset for point P (Resolves Issue 25)
        p_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thought.svg", color=COLOR_P).scale(0.2)
        
        # Use ValueTracker for the angle of point P on the ellipse
        theta = ValueTracker(60 * DEGREES)
        
        def get_p_pos():
            t = theta.get_value()
            return center_pos + np.array([a * np.cos(t), b * np.sin(t), 0])
        
        p_icon.add_updater(lambda m: m.move_to(get_p_pos()))
        
        p_label = Text("P", font_size=20, color=COLOR_P)
        p_label.add_updater(lambda l: l.next_to(p_icon, UR, buff=0.05))
        
        self.play(FadeIn(p_icon, p_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: "We will track the distance from P to each focus."
        self.lecture[2].set_color(COLOR_GEN)
        
        # Draw segments from P to F1 and P to F2
        line_pf1 = Line(get_p_pos(), f1_pos, color=COLOR_FOCI, stroke_width=2)
        line_pf2 = Line(get_p_pos(), f2_pos, color=COLOR_FOCI, stroke_width=2)
        
        line_pf1.add_updater(lambda l: l.put_start_and_end_on(get_p_pos(), f1_pos))
        line_pf2.add_updater(lambda l: l.put_start_and_end_on(get_p_pos(), f2_pos))
        
        # Additional Geometry: Generator and T points
        v_dot = Dot(v_pos, color=COLOR_GEN, radius=0.06)
        v_label = Text("V", font_size=20, color=COLOR_GEN)
        # Position v_label using grid (Resolves Issue 35)
        v_label_pos = (self.grid["A3"] + self.grid["A4"]) / 2
        v_label.move_to(v_label_pos).scale(0.8)
        
        def get_gen_unit_vec():
            p = get_p_pos()
            vec = p - v_pos
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else RIGHT
            
        def get_t1_pos():
            return v_pos + get_gen_unit_vec() * vt1
            
        def get_t2_pos():
            return v_pos + get_gen_unit_vec() * vt2

        # The generator line
        gen_line = Line(v_pos, get_t2_pos(), color=COLOR_GEN, stroke_width=2)
        gen_line.add_updater(lambda l: l.put_start_and_end_on(v_pos, get_t2_pos()))
        
        # Identify points T1 and T2
        t1_dot = Dot(get_t1_pos(), color=COLOR_GEN, radius=0.05)
        t2_dot = Dot(get_t2_pos(), color=COLOR_GEN, radius=0.05)
        t1_label = Text("T1", font_size=18, color=COLOR_GEN)
        t2_label = Text("T2", font_size=18, color=COLOR_GEN)
        
        t1_dot.add_updater(lambda d: d.move_to(get_t1_pos()))
        t2_dot.add_updater(lambda d: d.move_to(get_t2_pos()))
        t1_label.add_updater(lambda l: l.next_to(t1_dot, RIGHT, buff=0.1))
        t2_label.add_updater(lambda l: l.next_to(t2_dot, RIGHT, buff=0.1))
        
        self.play(Create(line_pf1), Create(line_pf2))
        self.play(FadeIn(v_dot, v_label))
        self.play(Create(gen_line))
        self.play(FadeIn(t1_dot, t2_dot, t1_label, t2_label))
        self.wait(1)
        
        # Conceptual movement along the ellipse
        self.play(theta.animate.set_value(135 * DEGREES), run_time=3)
        self.wait(1)
        self.play(theta.animate.set_value(-45 * DEGREES), run_time=3)
        self.wait(2)
