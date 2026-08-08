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
        lecture_lines = [
            "High-dimensional volume behaves in counter-intuitive ways.",
            "Most volume stays extremely close to the surface.",
            "Think of peeling a 1,000-dimensional orange.",
            "A thin peel contains nearly all the fruit's mass.",
            "Volume also concentrates heavily around the equator."
        ]
        self.setup_layout("Concentration of Measure", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # L1: High-dimensional volume behaves in counter-intuitive ways.
        # A1: Show 3D sphere split into shell #FFFF00 and core #808080. Area B3 to E6.
        self.lecture[0].set_color("#FFFF00")
        
        core = Circle(radius=1.5, color="#808080", fill_opacity=0.8, stroke_width=0)
        shell = Annulus(inner_radius=1.4, outer_radius=1.5, color="#FFFF00", fill_opacity=1.0, stroke_width=0)
        sphere_group = VGroup(core, shell)
        self.place_in_area(sphere_group, 'B3', 'E6', scale_factor=1.0)
        
        self.play(FadeIn(sphere_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # L2: Most volume stays extremely close to the surface.
        # A2: Visual of 'Orange' [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg] analogy. Area B3 to E6.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg]
        orange_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg")
        self.place_in_area(orange_svg, 'B3', 'E6', scale_factor=2.2)
        
        self.play(
            FadeOut(sphere_group),
            FadeIn(orange_svg)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # L3: Think of peeling a 1,000-dimensional orange.
        # A3: Label shell '99.9% Volume' in area B6. Shell glows #FFFF00.
        # Fix for Issue 35: Use B2-D6 for shell_label positioning to center it better relative to the volume.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        # Glow effect for orange "shell"
        shell_glow = Annulus(inner_radius=1.4, outer_radius=1.55, color="#FFFF00", fill_opacity=0.4, stroke_width=0)
        shell_glow.move_to(orange_svg.get_center())
        
        shell_label = Text("99.9% Volume", color="#FFFF00", font_size=24)
        # Applying VideoCritic fix for Issue 35
        self.place_in_area(shell_label, 'B2', 'D6', scale_factor=0.9)
        
        self.play(
            FadeIn(shell_glow),
            Write(shell_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # L4: A thin peel contains nearly all the fruit's mass.
        # A4: Highlight narrow belt around the sphere's equator in #00FFFF. Area B3 to E6.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#00FFFF")
        
        # Equator belt visualization
        equator_belt = Rectangle(height=0.4, width=3.0, color="#00FFFF", fill_opacity=0.6, stroke_width=0)
        equator_belt.move_to(orange_svg.get_center())
        
        self.play(
            FadeOut(shell_glow),
            FadeOut(shell_label),
            FadeIn(equator_belt)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # L5: Volume also concentrates heavily around the equator.
        # A5: Shrink belt width while intensifying color #00FFFF. Area B3 to E6.
        # Fix for Issue 34: self.place_in_area(equator_label, 'E2', 'F6', scale_factor=0.8)
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#00FFFF")
        
        equator_label = Text("Equator Concentration", color="#00FFFF", font_size=24)
        # Applying VideoCritic fix for Issue 34
        self.place_in_area(equator_label, 'E2', 'F6', scale_factor=0.8)
        
        self.play(
            equator_belt.animate.scale(0.5, about_point=equator_belt.get_center()).set_fill(opacity=1.0),
            Write(equator_label)
        )
        self.wait(2)
        
        # Final cleanup
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
