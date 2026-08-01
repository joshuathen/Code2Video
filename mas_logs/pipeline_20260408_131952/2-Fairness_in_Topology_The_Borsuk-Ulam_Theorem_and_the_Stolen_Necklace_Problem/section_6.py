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
        # Updated Lecture Lines from requirements
        self.setup_layout(
            "Conclusion and Real-World Intuition", 
            [
                'For k bead types, only k cuts are required.', 
                'Topology provides an elegant solution to fair resource allocation.', 
                'The Borsuk-Ulam Theorem makes the impossible division possible.'
            ]
        )

        # Colors
        COLOR_K = "#FFFFFF"
        COLOR_ALICE = "#FFC0CB" # Pink
        COLOR_BOB = "#87CEEB"   # Sky Blue
        COLOR_SPHERE = "#4444FF"
        COLOR_HIGHLIGHT = YELLOW

        # === Animation for Lecture Line 1 ===
        # Line: "For k bead types, only k cuts are required."
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        summary_text = Text("k beads, k cuts", weight=BOLD, color=COLOR_K)
        # Fix Issue 40: scale_factor from 1.2 to 1.0
        self.place_in_area(summary_text, "A2", "B5", scale_factor=1.0)
        
        self.play(Write(summary_text))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Line: "Topology provides an elegant solution to fair resource allocation."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)

        # Create a sphere and a necklace overlay
        sphere = Circle(radius=1.2, color=COLOR_SPHERE).set_stroke(opacity=0.5)
        sphere_glow = Circle(radius=1.2, color=COLOR_SPHERE).set_fill(COLOR_SPHERE, opacity=0.1)
        
        # Adding a grid-like structure to the sphere for "geometric grid" intuition
        grid_lines = VGroup(*[
            Line(sphere.get_top(), sphere.get_bottom(), stroke_width=1, color=COLOR_SPHERE).set_opacity(0.3).shift(RIGHT * x)
            for x in np.linspace(-0.8, 0.8, 5)
        ])
        
        # Necklace on sphere
        necklace_loop = Ellipse(width=2.4, height=0.6, color=WHITE).set_stroke(width=2)
        bead_colors = [RED, YELLOW, GREEN, BLUE, ORANGE, PURPLE]
        beads = VGroup(*[
            Dot(necklace_loop.point_from_proportion(i/6), color=bead_colors[i], radius=0.08)
            for i in range(6)
        ])
        
        geom_group = VGroup(sphere, sphere_glow, grid_lines, necklace_loop, beads)
        # Fix Issue 41: position from 'C1', 'D6' to 'C2', 'D5', scale from 0.9 to 0.8
        self.place_in_area(geom_group, "C2", "D5", scale_factor=0.8)
        
        self.play(Create(sphere), FadeIn(sphere_glow), Create(grid_lines))
        self.play(Create(necklace_loop), FadeIn(beads))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Line: "The Borsuk-Ulam Theorem makes the impossible division possible."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)

        # Fix Issue 48: Integrate assets for Alice, Bob, and Necklace
        alice_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/person.svg").set_color(COLOR_ALICE)
        alice_label = Text("Alice", font_size=20).next_to(alice_icon, DOWN, buff=0.1)
        alice_group = VGroup(alice_icon, alice_label)

        bob_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/person.svg").set_color(COLOR_BOB)
        bob_label = Text("Bob", font_size=20).next_to(bob_icon, DOWN, buff=0.1)
        bob_group = VGroup(bob_icon, bob_label)

        shared_necklace = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/necklace.svg").set_color(WHITE).set_stroke(width=2)
        # Satisfied smiles (added to icons)
        smile_a = Arc(radius=0.15, start_angle=-TAU/4 - 0.5, angle=1.0, color=WHITE).move_to(alice_icon.get_center()).shift(DOWN*0.1)
        smile_b = Arc(radius=0.15, start_angle=-TAU/4 - 0.5, angle=1.0, color=WHITE).move_to(bob_icon.get_center()).shift(DOWN*0.1)
        
        thieves_group = VGroup(alice_group, smile_a, shared_necklace, bob_group, smile_b).arrange(RIGHT, buff=0.4)
        
        # Fix Issue 42: position from 'E1', 'F6' to 'E2', 'F5', scale from 0.8 to 0.7
        self.place_in_area(thieves_group, "E2", "F5", scale_factor=0.7)

        self.play(FadeIn(thieves_group))
        self.wait(2)

        # Final shot cohesion
        summary_link = Arrow(geom_group.get_bottom(), thieves_group.get_top(), color=GREY_A, buff=0.2)
        self.play(Create(summary_link))
        self.wait(2)

        # Clean up highlights for final shot
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(3)
