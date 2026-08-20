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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Dandelin spheres connect solids to geometry.", "These tangency points are the foci.", "Physics relies on these for planetary orbits."]
        self.setup_layout("Summary and Synthesis", lecture_lines)
        
        # Mobjects for animations
        ellipse = Ellipse(width=2.0, height=1.2, color=BLUE_A)
        parabola = VGroup(
            Axes(x_range=[-1, 1], y_range=[0, 1], x_length=2, y_length=1.2, axis_config={"include_ticks": False}),
            FunctionGraph(lambda x: x**2, x_range=[-1, 1], color=GREEN_A)
        )
        hyperbola = VGroup(
            Axes(x_range=[-1, 1], y_range=[-1, 1], x_length=2, y_length=1.2, axis_config={"include_ticks": False}),
            FunctionGraph(lambda x: 1/x if x != 0 else None, x_range=[0.2, 1], color=RED_A),
            FunctionGraph(lambda x: 1/x if x != 0 else None, x_range=[-1, -0.2], color=RED_A)
        )
        
        # Assets
        planet_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg")
        orbit_overlay = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orbit.svg")

        # Labels
        label_e = Text("Ellipse", font_size=20)
        label_p = Text("Parabola", font_size=20)
        label_h = Text("Hyperbola", font_size=20)
        
        # Positioning
        self.place_at_grid(ellipse, 'B1', scale_factor=0.6)
        self.place_at_grid(parabola, 'B3', scale_factor=0.6)
        self.place_at_grid(hyperbola, 'B5', scale_factor=0.6)
        
        self.place_at_grid(label_e, 'C1', scale_factor=0.5)
        self.place_at_grid(label_p, 'C3', scale_factor=0.5)
        self.place_at_grid(label_h, 'C5', scale_factor=0.5)
        
        self.place_at_grid(planet_icon, 'A1', scale_factor=0.3)
        self.place_at_grid(orbit_overlay, 'B1', scale_factor=0.2)
        
        group_shapes = VGroup(ellipse, parabola, hyperbola, label_e, label_p, label_h, planet_icon, orbit_overlay)
        # Using the area layout per instructions
        self.place_in_area(group_shapes, 'B1', 'D6', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE_A), Create(ellipse), FadeIn(planet_icon), FadeIn(label_e))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN_A), Create(parabola), FadeIn(label_p))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED_A), Create(hyperbola), FadeIn(label_h), FadeIn(orbit_overlay))
        
        self.wait(2)
        
        # Fade out
        self.play(FadeOut(self.lecture), FadeOut(group_shapes))
        
        # Final Text
        summary = Text("Dandelin's Principle: From 3D to Planetary Orbits", font_size=32)
        self.play(Write(summary))
        self.wait(3)
