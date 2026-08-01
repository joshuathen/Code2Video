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
        lecture_lines = [
            "Atoms have natural resonant frequencies for electron vibration.",
            "Light closer to resonance interacts more strongly.",
            "Higher frequencies like violet create larger phase lags.",
            "This causes violet light to bend more than red.",
            "This frequency-dependent bending is known as dispersion."
        ]
        self.setup_layout("Color Dependence: The Resonance Effect", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Atoms have natural resonant frequencies for electron vibration.
        # Display a graph of refractive index 'n' vs wavelength 'lambda' (#FFFFFF) showing normal dispersion.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        axes = Axes(
            x_range=[400, 750, 100],
            y_range=[1.4, 1.7, 0.1],
            axis_config={"include_tip": True, "color": WHITE},
            x_length=4,
            y_length=3,
        )
        # Using Text for axis labels
        x_label = axes.get_x_axis_label(Text("λ", font_size=24), edge=DOWN, direction=DOWN, buff=0.1)
        y_label = axes.get_y_axis_label(Text("n", font_size=24, slant=ITALIC), edge=LEFT, direction=LEFT, buff=0.1)
        
        # Cauchy Equation curve
        graph = axes.plot(
            lambda x: 1.45 + (10000 / x**2) * 5, 
            x_range=[400, 700], 
            color=WHITE
        )
        graph_group = VGroup(axes, x_label, y_label, graph)
        # Fix for Issue 41: Reposition to avoid title overlap
        self.place_in_area(graph_group, "B1", "D6", scale_factor=0.8)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(graph))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Light closer to resonance interacts more strongly.
        # Show a red wave (#FF0000) and a violet wave (#8F00FF) both approaching a row of atoms.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Atom Row
        atoms = VGroup(*[Circle(radius=0.15, color=BLUE, fill_opacity=0.3) for _ in range(2)])
        atoms.arrange(DOWN, buff=1.0)
        
        electrons = VGroup(*[Dot(color="#FFFF00", radius=0.08) for _ in range(2)])
        for i in range(2):
            electrons[i].move_to(atoms[i].get_center())
            
        # Waves approaching
        red_wave = FunctionGraph(
            lambda x: 0.25 * np.sin(3 * x),
            x_range=[-1.5, 0],
            color=RED
        ).next_to(atoms[0], LEFT, buff=0.5)
        
        violet_wave = FunctionGraph(
            lambda x: 0.25 * np.sin(7 * x),
            x_range=[-1.5, 0],
            color="#8F00FF"
        ).next_to(atoms[1], LEFT, buff=0.5)

        # Fix for Issue 40: Group elements and use place_in_area for better composition
        atom_interaction_group = VGroup(atoms, electrons, red_wave, violet_wave)
        self.place_in_area(atom_interaction_group, "E1", "F6", scale_factor=0.9)

        self.play(Create(atoms), Create(electrons))
        self.play(Create(red_wave), Create(violet_wave))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Higher frequencies like violet create larger phase lags.
        # Animate the electrons (#FFFF00) vibrating more intensely in response to the violet wave's frequency.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Vibration animation: Violet (bottom) vibrates more intensely
        self.play(
            electrons[0].animate(rate_func=wiggle).shift(UP*0.05),
            electrons[1].animate(rate_func=wiggle).shift(UP*0.18),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This causes violet light to bend more than red.
        # Show the violet wave experiencing a larger phase lag compared to the red wave.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Waves exit with lag (positioned relative to the atoms placed earlier)
        red_wave_exit = FunctionGraph(
            lambda x: 0.25 * np.sin(3 * x),
            x_range=[0, 1.5],
            color=RED
        ).next_to(atoms[0], RIGHT, buff=0.5)
        
        # Lagged violet wave (shifted right/phase delayed)
        violet_wave_exit = FunctionGraph(
            lambda x: 0.25 * np.sin(7 * (x - 0.4)), 
            x_range=[0, 1.5],
            color="#8F00FF"
        ).next_to(atoms[1], RIGHT, buff=0.5)

        self.play(
            Create(red_wave_exit),
            Create(violet_wave_exit),
            red_wave.animate.set_opacity(0.4),
            violet_wave.animate.set_opacity(0.4)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # This frequency-dependent bending is known as dispersion.
        # Visualize violet light bending more than red light when passing through a prism.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Clear previous elements
        self.play(
            FadeOut(graph_group),
            FadeOut(atom_interaction_group),
            FadeOut(red_wave_exit),
            FadeOut(violet_wave_exit)
        )
        
        # Fix for Issue 32: Use prism asset
        prism = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/prism.svg")
        prism.set_color(WHITE)
        # Fix for Issue 42: Adjusted scale factor for ray breathing room
        self.place_in_area(prism, "B2", "E5", scale_factor=0.75)
        
        # Light rays through prism (schematic representation)
        entry_pt = prism.get_left() + RIGHT * 0.1
        white_ray = Line(start=entry_pt + LEFT * 2, end=entry_pt, color=WHITE)
        
        # Red exits with slight downward tilt
        red_exit_end = prism.get_right() + RIGHT * 1.5 + DOWN * 0.3
        red_ray = Line(start=entry_pt, end=red_exit_end, color=RED)
        
        # Violet exits with steeper downward tilt
        violet_exit_end = prism.get_right() + RIGHT * 1.5 + DOWN * 0.8
        violet_ray = Line(start=entry_pt, end=violet_exit_end, color="#8F00FF")
        
        self.play(Create(prism))
        self.play(Create(white_ray))
        self.play(Create(red_ray), Create(violet_ray))
        
        self.wait(3)
