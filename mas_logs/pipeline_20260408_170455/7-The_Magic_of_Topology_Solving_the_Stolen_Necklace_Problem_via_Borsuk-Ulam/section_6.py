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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Conclusion: The Power of Abstraction", 
            [
                "Topology turns a discrete puzzle into a geometric solution.", 
                "Simple properties of spheres solve complex division problems.", 
                "This is the hidden power of mathematical abstraction."
            ]
        )

        # Colors for highlighting lecture lines
        highlight_color = YELLOW

        # === Animation for Lecture Line 1 ===
        # Visual: The necklace circle morphs into a smooth sphere, then into 'Topology'
        self.lecture[0].set_color(highlight_color)
        
        # 1. Create necklace (ring of beads)
        necklace = VGroup(*[Dot(radius=0.08, color=interpolate_color(RED, BLUE, i/11)) 
                          for i in range(12)])
        
        # Circular arrangement for the necklace
        for i, dot in enumerate(necklace):
            angle = i * (2 * PI / 12)
            dot.move_to(np.array([np.cos(angle), np.sin(angle), 0]))
        
        self.place_in_area(necklace, "B3", "D4", scale_factor=0.8)
        self.play(FadeIn(necklace))
        self.wait(1)

        # 2. Morph to smooth sphere
        sphere = Circle(radius=1.0, color=BLUE_B, fill_opacity=0.3)
        sphere_mesh1 = Ellipse(width=2.0, height=0.6, color=BLUE_D).move_to(sphere)
        sphere_mesh2 = Line(UP, DOWN, color=BLUE_D).scale(1.0).move_to(sphere)
        sphere_group = VGroup(sphere, sphere_mesh1, sphere_mesh2)
        self.place_in_area(sphere_group, "B3", "D4", scale_factor=0.8)

        self.play(ReplacementTransform(necklace, sphere_group))
        self.wait(1)

        # 3. Morph to 'Topology' text
        topology_text = Text("Topology", font_size=36, color="#00FFFF")
        self.place_at_grid(topology_text, "C4", scale_factor=1.0)
        
        self.play(ReplacementTransform(sphere_group, topology_text))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Visual: 'Discrete Problem' <-> 'Continuous Solution'
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(highlight_color)
        )
        
        self.play(FadeOut(topology_text))

        discrete_text = Text("Discrete\nProblem", font_size=24, color="#FF9999")
        continuous_text = Text("Continuous\nSolution", font_size=24, color="#99FF99")
        arrow = DoubleArrow(LEFT, RIGHT, color=WHITE, buff=0.2)

        self.place_in_area(discrete_text, "D1", "D2", scale_factor=0.8)
        self.place_in_area(arrow, "D3", "D4", scale_factor=0.6)
        self.place_in_area(continuous_text, "D5", "D6", scale_factor=0.8)

        self.play(
            FadeIn(discrete_text),
            GrowFromCenter(arrow),
            FadeIn(continuous_text)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Visual: The two thieves reappear, smiling and holding their equal shares of the necklace.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(highlight_color)
        )

        self.play(FadeOut(discrete_text), FadeOut(continuous_text), FadeOut(arrow))

        # Asset: [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/th.svg]
        thief1_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/th.svg").set_color(ORANGE)
        thief2_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/th.svg").set_color(PURPLE)
        
        # Add a smile for the "smiling" description
        smile1 = Arc(radius=0.1, start_angle=PI, angle=PI, color=WHITE).shift(UP*0.1)
        smile2 = Arc(radius=0.1, start_angle=PI, angle=PI, color=WHITE).shift(UP*0.1)
        
        thief1 = VGroup(thief1_icon, smile1)
        thief2 = VGroup(thief2_icon, smile2)
        
        # Equal shares of the necklace (simplified as two colored arcs)
        share1 = Arc(radius=0.4, start_angle=0, angle=PI, color="#FFD700")
        share2 = Arc(radius=0.4, start_angle=PI, angle=PI, color="#C0C0C0")

        self.place_at_grid(thief1, "B2", scale_factor=0.8)
        self.place_at_grid(thief2, "B5", scale_factor=0.8)
        self.place_at_grid(share1, "C2", scale_factor=0.8)
        self.place_at_grid(share2, "C5", scale_factor=0.8)

        self.play(
            FadeIn(thief1),
            FadeIn(thief2),
            Create(share1),
            Create(share2)
        )
        self.wait(3)

        # Final cleanup highlight
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
