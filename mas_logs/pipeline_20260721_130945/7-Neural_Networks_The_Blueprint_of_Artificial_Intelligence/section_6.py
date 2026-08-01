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

class Section6Scene(TeachingScene):
    def construct(self):
        # setup
        title_text = "Learning via Feedback: The Error Correction"
        lecture_lines = [
            "After a guess, the network compares it to the truth.",
            "A loss function calculates the distance between guess and reality.",
            "Backpropagation identifies which weights caused the specific error.",
            "The network wiggles weights to minimize the calculated loss.",
            "Through repeated feedback, Pixel correctly identifies the lemon."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_BALL = BLUE_B
        COLOR_LEMON = YELLOW_B
        COLOR_ERROR = "#FF0000"
        COLOR_SUCCESS = GREEN_B
        COLOR_HIGHLIGHT = ORANGE
        
        # Assets
        LEMON_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/lemon.svg"
        BALL_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg"
        
        # === Animation for Lecture Line 1 ===
        # After a guess, the network compares it to the truth.
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        # Consistent with lecture line 5 ("identifies the lemon"), Truth is Lemon.
        # Initial incorrect guess is Ball.
        
        # Use provided assets (Issue 20)
        ball_icon = SVGMobject(BALL_PATH).set_color(COLOR_BALL).scale(0.3)
        guess_label = VGroup(Text("Guess:", font_size=20), ball_icon).arrange(RIGHT, buff=0.1)
        
        lemon_icon = SVGMobject(LEMON_PATH).set_color(COLOR_LEMON).scale(0.3)
        truth_label = VGroup(Text("Truth:", font_size=20), lemon_icon).arrange(RIGHT, buff=0.1)
        
        # Position fixes (Issues 36, 37, 38)
        self.place_at_grid(guess_label, 'B3', scale_factor=0.8)
        self.place_at_grid(truth_label, 'C3', scale_factor=0.8)
        
        cross = Cross(stroke_color=COLOR_ERROR, stroke_width=8).scale(0.3)
        self.place_at_grid(cross, 'B4', scale_factor=1.0)
        
        self.play(FadeIn(guess_label), FadeIn(truth_label))
        self.play(Create(cross))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A loss function calculates the distance between guess and reality.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_ERROR)
        
        loss_tracker = ValueTracker(0.85)
        loss_text = Text("Loss:", color=WHITE, font_size=22)
        loss_val = DecimalNumber(loss_tracker.get_value(), color=COLOR_ERROR, num_decimal_places=2)
        loss_val.add_updater(lambda d: d.set_value(loss_tracker.get_value()))
        
        loss_group = VGroup(loss_text, loss_val).arrange(RIGHT, buff=0.2)
        self.place_at_grid(loss_group, "B5", scale_factor=0.7)
        
        self.play(FadeIn(loss_group))
        self.play(Indicate(loss_group)) # L004
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Backpropagation identifies which weights caused the specific error.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Network Construction
        input_nodes = VGroup(*[Circle(radius=0.12, color=WHITE) for _ in range(2)])
        hidden_nodes = VGroup(*[Circle(radius=0.12, color=WHITE) for _ in range(3)])
        output_nodes = VGroup(*[Circle(radius=0.12, color=WHITE) for _ in range(2)])
        
        input_nodes.arrange(DOWN, buff=0.5)
        hidden_nodes.arrange(DOWN, buff=0.4)
        output_nodes.arrange(DOWN, buff=0.5)
        
        net_layers = VGroup(input_nodes, hidden_nodes, output_nodes).arrange(RIGHT, buff=1.2)
        self.place_in_area(net_layers, "D2", "F6", scale_factor=0.8) # Area-Positioning Rule
        
        connections = VGroup()
        for i_node in input_nodes:
            for h_node in hidden_nodes:
                connections.add(Line(i_node.get_center(), h_node.get_center(), stroke_width=2, color=GRAY))
        for h_node in hidden_nodes:
            for o_node in output_nodes:
                connections.add(Line(h_node.get_center(), o_node.get_center(), stroke_width=2, color=GRAY))
        
        self.play(Create(net_layers), Create(connections))
        
        # Highlight problematic weights
        bad_weights = VGroup(connections[0], connections[4], connections[8])
        self.play(bad_weights.animate.set_color(COLOR_ERROR).set_stroke(width=5))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The network wiggles weights to minimize the calculated loss.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_SUCCESS)
        
        # Wiggle logic using renderer time (L008)
        for bw in bad_weights:
            bw.initial_pos = bw.get_center().copy()
            bw.add_updater(lambda m: m.move_to(m.initial_pos + 0.05 * np.sin(self.renderer.time * 20) * RIGHT))
            
        # Animate loss decreasing and random weight changes
        self.play(
            loss_tracker.animate.set_value(0.12),
            *[c.animate.set_stroke(width=np.random.uniform(1, 4)) for c in connections if c not in bad_weights],
            run_time=3
        )
        for bw in bad_weights:
            bw.clear_updaters()
            bw.move_to(bw.initial_pos)
        
        self.play(bad_weights.animate.set_color(GRAY).set_stroke(width=2)) 
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Through repeated feedback, Pixel correctly identifies the lemon.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_SUCCESS)
        
        # Final success: Guess becomes Lemon
        new_lemon_icon = SVGMobject(LEMON_PATH).set_color(COLOR_SUCCESS).scale(0.3)
        final_guess = VGroup(Text("Guess:", font_size=20), new_lemon_icon).arrange(RIGHT, buff=0.1)
        self.place_at_grid(final_guess, 'B3', scale_factor=0.8)
        
        check_mark = Text("✔", color=COLOR_SUCCESS, font_size=32)
        self.place_at_grid(check_mark, 'B4', scale_factor=1.0)
        
        self.play(
            FadeOut(cross),
            ReplacementTransform(guess_label, final_guess),
            truth_label.animate.set_color(COLOR_SUCCESS),
            FadeIn(check_mark)
        )
        self.play(Indicate(final_guess), Indicate(check_mark))
        self.wait(3)
