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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "Visualizing the Magnitude of 2^256"
        lines = [
            "$2^{256}$ is roughly $1.15$ times $10^{77}$ combinations.",
            "That is close to the atoms in the universe.",
            "Imagine searching for one specific atom in space.",
            "The scale spans from sand grains to entire galaxies.",
            "Finding one specific hash is practically impossible."
        ]
        self.setup_layout(title, lines)

        # Colors
        WHITE_COLOR = "#FFFFFF"
        SAND_COLOR = "#FFFFE0"
        EARTH_COLOR = "#ADD8E6"
        GALAXY_COLOR = "#FFD700"
        ATOM_COLOR = "#FF0000"
        HIGHLIGHT_COLOR = YELLOW

        # Assets
        SAND_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sand.svg"
        EARTH_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/earth.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        math_expr = MathTex(r"2^{256} \approx 1.15 \times 10^{77}", color=WHITE_COLOR)
        self.place_in_area(math_expr, "A2", "B5", scale_factor=1.0)
        
        self.play(Write(math_expr))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Atom comparison label
        atom_label = Text("Total Atoms in Observable Universe: ~10^80", font_size=18, color=WHITE_COLOR)
        self.place_at_grid(atom_label, "D4", scale_factor=0.9)
        
        comparison_arrow = Arrow(math_expr.get_bottom(), atom_label.get_top(), color=WHITE_COLOR, buff=0.2)
        
        self.play(Create(comparison_arrow), Write(atom_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Clear for visualization
        self.play(FadeOut(math_expr), FadeOut(atom_label), FadeOut(comparison_arrow))
        
        # Grain of sand asset
        sand_svg = SVGMobject(SAND_ASSET, color=SAND_COLOR)
        sand_label = Text("Grain of Sand", font_size=16, color=SAND_COLOR)
        self.place_at_grid(sand_svg, "D3", scale_factor=0.3)
        self.place_at_grid(sand_label, "E3", scale_factor=0.8)
        
        self.play(DrawBorderThenFill(sand_svg), FadeIn(sand_label))
        self.wait(1)

        # Zoom out to beach (represented by a large area of small dots)
        beach_area = RoundedRectangle(width=3, height=2, corner_radius=0.2, color="#C2B280", fill_opacity=0.3)
        beach_label = Text("A Vast Beach", font_size=16, color="#C2B280")
        self.place_in_area(beach_area, "C2", "E5")
        self.place_at_grid(beach_label, "B2", scale_factor=0.8)
        
        self.play(
            sand_svg.animate.scale(0.2).move_to(beach_area.get_center()),
            FadeOut(sand_label),
            FadeIn(beach_area),
            Write(beach_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Zoom out to Earth asset
        earth_svg = SVGMobject(EARTH_ASSET, color=EARTH_COLOR)
        earth_label = Text("Planet Earth", font_size=16, color=EARTH_COLOR)
        self.place_in_area(earth_svg, "C2", "E5", scale_factor=1.2)
        self.place_at_grid(earth_label, "B2", scale_factor=0.8)
        
        self.play(
            FadeOut(beach_area),
            FadeOut(beach_label),
            FadeOut(sand_svg),
            FadeIn(earth_svg),
            Write(earth_label)
        )
        self.wait(1)
        
        # Zoom out to Galaxy
        galaxy_core = Dot(color=WHITE, radius=0.1)
        galaxy_spirals = VGroup(*[
            ParametricFunction(
                lambda t: np.array([0.5 * t * np.cos(t + i*PI/2), 0.3 * t * np.sin(t + i*PI/2), 0]),
                t_range=[0, 3*PI], color=GALAXY_COLOR, stroke_width=2
            ) for i in range(4)
        ])
        galaxy_vgroup = VGroup(galaxy_core, galaxy_spirals)
        galaxy_label = Text("Milky Way Galaxy", font_size=16, color=GALAXY_COLOR)
        self.place_in_area(galaxy_vgroup, "B1", "F6", scale_factor=0.8)
        self.place_at_grid(galaxy_label, "A1", scale_factor=0.8)
        
        self.play(
            FadeOut(earth_svg),
            FadeOut(earth_label),
            Create(galaxy_vgroup),
            Write(galaxy_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # A red dot flashing in the galaxy representing an atom/hash
        target_atom = Dot(color=ATOM_COLOR, radius=0.06)
        # Position it somewhere specific in the galaxy
        target_atom.move_to(galaxy_vgroup.get_center() + RIGHT * 1.2 + UP * 0.4)
        atom_pointer_label = Text("One Specific Atom (Hash)", font_size=16, color=ATOM_COLOR)
        self.place_at_grid(atom_pointer_label, "B5", scale_factor=0.8)
        
        arrow_to_atom = Arrow(atom_pointer_label.get_left(), target_atom.get_center(), color=ATOM_COLOR, buff=0.1)
        
        self.play(FadeIn(target_atom), Write(atom_pointer_label), Create(arrow_to_atom))
        
        # Flashing effect
        for _ in range(3):
            self.play(Flash(target_atom, color=ATOM_COLOR, flash_radius=0.15, run_time=0.5))
            self.wait(0.2)
            
        self.wait(2)
        self.lecture[4].set_color(WHITE)
