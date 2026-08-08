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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Defining the PDE", [
            "A PDE involves a function and its partial derivatives.",
            "Partial derivatives measure change along one specific direction.",
            "Imagine hiking on a 3D mountain landscape surface.",
            "One hiker moves North-South, measuring the vertical slope.",
            "Another moves East-West, tracking change in that direction."
        ])
        
        # Colors
        color_ns = "#FF00FF" # Magenta for partial u / partial y
        color_ew = "#00FF00" # Green for partial u / partial x
        surface_color = BLUE_D

        # Assets
        hiker_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Surface u(x, y) = 0.5*cos(x)*sin(y)
        surface = Surface(
            lambda u, v: np.array([u, v, 0.5 * np.cos(u) * np.sin(v)]),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(16, 16),
            should_make_jagged=False,
            fill_color=surface_color,
            fill_opacity=0.6,
            stroke_color=WHITE,
            stroke_width=0.5
        )
        # Apply rotation for 3D look in 2D scene
        surface.rotate(60 * DEGREES, axis=RIGHT)
        surface.rotate(-30 * DEGREES, axis=OUT)
        
        # Place the surface in area C2 to F6 (per Issue 35)
        self.place_in_area(surface, "C2", "F6", scale_factor=0.7)
        
        surface_label = MathTex("u(x, y)", color=WHITE)
        # Place at A6 (per Issue 33)
        self.place_at_grid(surface_label, "A6", scale_factor=0.8)

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW),
            Create(surface),
            Write(surface_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # North-South hiker: partial u / partial y (y-axis movement)
        # Load asset (Issue 26)
        hiker_ns = SVGMobject(hiker_asset_path).set_color(color_ns)
        hiker_ns.scale(0.15)
        
        label_ns = MathTex(r"\frac{\partial u}{\partial y}", color=color_ns)
        # Place at A2 (per Issue 34)
        self.place_at_grid(label_ns, "A2", scale_factor=0.8) 

        v_tracker = ValueTracker(-2)
        u_fixed = 0
        
        def update_hiker_ns(m):
            v = v_tracker.get_value()
            pos_3d = np.array([u_fixed, v, 0.5 * np.cos(u_fixed) * np.sin(v)])
            # Apply same manual rotation as surface
            rotated_pos = pos_3d.copy()
            y, z = rotated_pos[1], rotated_pos[2]
            rotated_pos[1] = y * np.cos(60*DEGREES) - z * np.sin(60*DEGREES)
            rotated_pos[2] = y * np.sin(60*DEGREES) + z * np.cos(60*DEGREES)
            x, y = rotated_pos[0], rotated_pos[1]
            rotated_pos[0] = x * np.cos(-30*DEGREES) - y * np.sin(-30*DEGREES)
            rotated_pos[1] = x * np.sin(-30*DEGREES) + y * np.cos(-30*DEGREES)
            m.move_to(surface.get_center() + rotated_pos * 0.7)

        hiker_ns.add_updater(update_hiker_ns)
        path_ns = TracedPath(hiker_ns.get_center, stroke_color=color_ns, stroke_width=4)

        self.add(path_ns)
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW),
            FadeIn(hiker_ns),
            Write(label_ns)
        )
        self.play(v_tracker.animate.set_value(2), run_time=3, rate_func=linear)
        hiker_ns.remove_updater(update_hiker_ns)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # East-West hiker: partial u / partial x (x-axis movement)
        # Load asset (Issue 26)
        hiker_ew = SVGMobject(hiker_asset_path).set_color(color_ew)
        hiker_ew.scale(0.15)
        
        label_ew = MathTex(r"\frac{\partial u}{\partial x}", color=color_ew)
        # Place at A4 (per Issue 34)
        self.place_at_grid(label_ew, "A4", scale_factor=0.8)

        u_tracker = ValueTracker(-2)
        v_fixed = 0.5
        
        def update_hiker_ew(m):
            u = u_tracker.get_value()
            pos_3d = np.array([u, v_fixed, 0.5 * np.cos(u) * np.sin(v_fixed)])
            rotated_pos = pos_3d.copy()
            y, z = rotated_pos[1], rotated_pos[2]
            rotated_pos[1] = y * np.cos(60*DEGREES) - z * np.sin(60*DEGREES)
            rotated_pos[2] = y * np.sin(60*DEGREES) + z * np.cos(60*DEGREES)
            x, y = rotated_pos[0], rotated_pos[1]
            rotated_pos[0] = x * np.cos(-30*DEGREES) - y * np.sin(-30*DEGREES)
            rotated_pos[1] = x * np.sin(-30*DEGREES) + y * np.cos(-30*DEGREES)
            m.move_to(surface.get_center() + rotated_pos * 0.7)

        hiker_ew.add_updater(update_hiker_ew)
        path_ew = TracedPath(hiker_ew.get_center, stroke_color=color_ew, stroke_width=4)

        self.add(path_ew)
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW),
            FadeIn(hiker_ew),
            Write(label_ew)
        )
        self.play(u_tracker.animate.set_value(2), run_time=3, rate_func=linear)
        hiker_ew.remove_updater(update_hiker_ew)
        self.wait(2)

        # Final fade out of visuals
        self.play(
            *[FadeOut(m) for m in self.mobjects if m != self.title and m != self.lecture]
        )
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
