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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup initial layout
        lines = [
            "How do we trace infections without tracking people?",
            "Traditional tracing creates a central map of movements.",
            "DP-3T offers a decentralized, privacy-first alternative."
        ]
        self.setup_layout("The Privacy Dilemma (0:45)", lines)

        # === Animation for Lecture Line 1 ===
        # Colors line 1
        self.lecture[0].set_color(WHITE)
        
        # Create Alice and Bob dots
        alice = Dot(color="#3498DB")
        bob = Dot(color="#E67E22")
        alice_label = Text("Alice", font_size=16, color="#3498DB")
        bob_label = Text("Bob", font_size=16, color="#E67E22")

        # Initial placement - Visual fixes applied (Issues 28, 29)
        self.place_at_grid(alice, "D3", scale_factor=0.8)
        self.place_at_grid(bob, "D5", scale_factor=0.8)
        alice_label.next_to(alice, UP, buff=0.1)
        bob_label.next_to(bob, UP, buff=0.1)

        self.play(
            FadeIn(alice), FadeIn(bob),
            FadeIn(alice_label), FadeIn(bob_label)
        )

        # Alice and Bob wander
        self.play(
            alice.animate.move_to(self.grid["B2"]),
            alice_label.animate.next_to(self.grid["B2"], UP, buff=0.1),
            bob.animate.move_to(self.grid["E5"]),
            bob_label.animate.next_to(self.grid["E5"], UP, buff=0.1),
            run_time=2,
            rate_func=linear
        )

        # === Animation for Lecture Line 2 ===
        # Highlight line 2 in Red to match the eye
        self.lecture[1].set_color("#E74C3C")
        
        # Issue 25: Use asset for the central "Eye" server
        eye_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/eye.svg")
        eye_asset.set_color("#E74C3C")
        self.place_at_grid(eye_asset, "C4", scale_factor=0.8)

        self.play(FadeIn(eye_asset))
        
        # Eye "watches" Alice and Bob
        self.play(
            eye_asset.animate.shift(LEFT * 0.1),
            alice.animate.move_to(self.grid["B1"]),
            alice_label.animate.next_to(self.grid["B1"], UP, buff=0.1),
            run_time=1.5
        )
        self.play(
            eye_asset.animate.shift(RIGHT * 0.2),
            bob.animate.move_to(self.grid["F6"]),
            bob_label.animate.next_to(self.grid["F6"], UP, buff=0.1),
            run_time=1.5
        )

        # === Animation for Lecture Line 3 ===
        # Highlight line 3 in Blue to match the shield
        self.lecture[2].set_color("#3498DB")
        
        # Privacy Shield replaces the eye
        shield_shape = VMobject()
        shield_shape.set_points_as_corners([
            [-0.5, 0.6, 0], [0.5, 0.6, 0], [0.5, 0, 0], [0, -0.7, 0], [-0.5, 0, 0], [-0.5, 0.6, 0]
        ])
        shield = shield_shape.set_fill("#3498DB", opacity=0.8).set_stroke(WHITE, width=2)
        self.place_at_grid(shield, "C4", scale_factor=0.8)

        self.play(FadeOut(eye_asset), FadeIn(shield))
        
        # Alice and Bob move to exchange positions
        pos_a_xch = self.grid["D2"]
        pos_b_xch = self.grid["D5"]
        
        self.play(
            alice.animate.move_to(pos_a_xch),
            alice_label.animate.next_to(pos_a_xch, UP, buff=0.1),
            bob.animate.move_to(pos_b_xch),
            bob_label.animate.next_to(pos_b_xch, UP, buff=0.1),
            run_time=1.5
        )
        
        # Alice and Bob exchange Ephemeral IDs (Glowing yellow dots)
        id_a = Dot(radius=0.1, color="#F1C40F")
        id_b = Dot(radius=0.1, color="#F1C40F")
        id_a.move_to(alice.get_center())
        id_b.move_to(bob.get_center())
        
        self.play(FadeIn(id_a), FadeIn(id_b))
        
        # Paths for exchange
        self.play(
            id_a.animate.move_to(bob.get_center()),
            id_b.animate.move_to(alice.get_center()),
            run_time=1.5
        )
        
        self.play(FadeOut(id_a), FadeOut(id_b))

        # Final title "DP-3T"
        dp3t_title = Text("DP-3T", font_size=42, color=WHITE)
        # Visual fix (Issue 30): Scale factor 0.8
        self.place_at_grid(dp3t_title, "C4", scale_factor=0.8)
        
        self.play(
            FadeOut(shield),
            FadeIn(dp3t_title)
        )
        self.wait(2)
