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
        # Setup layout with updated script
        lines = [
            "Domain coloring maps complex outputs to specific colors.",
            "The phase sets the hue on a color wheel.",
            "Magnitude sets brightness; darker areas are closer to zero.",
            "Function roots appear as vibrant, rainbow-colored pinwheels.",
            "All colors converge at the exact location of roots."
        ]
        self.setup_layout("Prerequisite: Visualizing Functions with Domain Coloring", lines)

        def create_domain_map(func, res=100):
            # Domain bounds [-2, 2]
            x = np.linspace(-2, 2, res)
            y = np.linspace(-2, 2, res)
            X, Y = np.meshgrid(x, y)
            Z = X + 1j * Y
            W = func(Z)
            
            angles = np.angle(W)
            hues = (angles + np.pi) / (TAU)
            mags = np.abs(W)
            # Use a slightly more dramatic brightness mapping for roots
            vals = np.clip(mags, 0, 1)
            
            # Vectorized HSV to RGB
            h6 = hues * 6
            idx = h6.astype(int)
            f = h6 - idx
            p = 0.0
            q = vals * (1.0 - f)
            t = vals * f
            
            r = np.choose(idx % 6, [vals, q, p, p, t, vals])
            g = np.choose(idx % 6, [t, vals, vals, q, p, p])
            b = np.choose(idx % 6, [p, p, t, vals, vals, q])
            
            rgb = np.stack([r, g, b], axis=-1)
            rgb = (rgb * 255).astype(np.uint8)
            img = ImageMobject(np.flipud(rgb))
            img.height = 4.0 # Fits well in B-F area
            return img

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00") 
        
        plane = ComplexPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1],
            background_line_style={"stroke_color": "#FFFFFF", "stroke_opacity": 0.4}
        ).scale(1.0) # Scale adjusted to fit within B1-F6 height
        self.place_in_area(plane, 'B1', 'F6')
        
        map_identity = create_domain_map(lambda z: z)
        self.place_in_area(map_identity, 'B1', 'F6')
        
        self.play(Create(plane))
        self.play(FadeIn(map_identity), plane.animate.set_stroke(opacity=0.2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color("#FFFFFF"),
            self.lecture[1].animate.set_color("#FFFF00")
        )

        # Color Wheel Legend (Issue 30 fixes)
        wheel = VGroup()
        colors = ["#FF0000", "#FFFF00", "#00FF00", "#00FFFF", "#0000FF", "#FF00FF"]
        for i in range(12):
            sector = AnnularSector(
                inner_radius=0.3, outer_radius=0.6,
                angle=TAU/12, start_angle=i*TAU/12,
                color=colors[i // 2 % 6]
            )
            wheel.add(sector)
        
        self.place_at_grid(wheel, 'C5', scale_factor=1.1)
        wheel_label = Text("Phase = Hue", font_size=18, color="#FFFFFF")
        self.place_at_grid(wheel_label, 'B5', scale_factor=0.9)
        
        self.play(FadeIn(wheel), Write(wheel_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color("#FFFFFF"),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        # Highlight the center (brightness -> 0)
        origin_dot = Dot(plane.n2p(0), color="#FFFFFF")
        brightness_label = Text("Darkness = Magnitude 0", font_size=16, color="#FFFFFF")
        self.place_at_grid(brightness_label, 'F1')
        
        self.play(Indicate(origin_dot), Write(brightness_label))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color("#FFFFFF"),
            self.lecture[3].animate.set_color("#FFFF00")
        )

        # f(z) = z^2 - 1 (Issue 28 and 29 fixes)
        map_z2 = create_domain_map(lambda z: z**2 - 1)
        self.place_in_area(map_z2, 'B1', 'F6', scale_factor=1.0)
        
        formula = Text("f(z) = z^2 - 1", font_size=24, color="#FFFFFF")
        self.place_at_grid(formula, 'A3', scale_factor=1.2)
        
        self.play(
            ReplacementTransform(map_identity, map_z2),
            Write(formula),
            FadeOut(wheel),
            FadeOut(wheel_label),
            FadeOut(brightness_label)
        )

        # Roots at z = 1 and z = -1
        root1 = Dot(plane.n2p(1 + 0j), color="#FFFFFF", radius=0.08)
        root2 = Dot(plane.n2p(-1 + 0j), color="#FFFFFF", radius=0.08)
        
        self.play(Flash(root1, color="#FFFFFF"), Flash(root2, color="#FFFFFF"))
        self.add(root1, root2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color("#FFFFFF"),
            self.lecture[4].animate.set_color("#FFFF00")
        )

        # Zoom into one pinwheel (root at z=1)
        # Shift slightly to keep it in grid area
        display_center = self.grid['D3'] 
        shift_vector = display_center - root1.get_center()
        
        visualization_group = Group(plane, map_z2, root1, root2)
        
        self.play(
            visualization_group.animate.shift(shift_vector).scale(1.8, about_point=display_center),
            formula.animate.set_color("#888888"),
            run_time=2
        )
        self.wait(2)

        # Finalize
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        self.wait(1)
