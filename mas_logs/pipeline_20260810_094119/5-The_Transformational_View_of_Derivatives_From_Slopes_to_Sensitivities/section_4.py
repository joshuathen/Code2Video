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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The derivative unifies graphical slopes and transformations.",
            "Growth_Tree visualizes the function's steady progress.",
            "Growth_Rhythm plots the derivative's speed over time."
        ]
        self.setup_layout("Synthesis & Visual Summary", lecture_lines)
        
        # --- Create Assets ---
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/tree.svg]
        tree = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tree.svg")
        
        rhythm_graph = Axes(x_range=[0, 5, 1], y_range=[0, 2, 0.5], axis_config={"include_tip": False})
        curve = rhythm_graph.plot(lambda x: 0.1*x**2, color=BLUE)
        graph_group = VGroup(rhythm_graph, curve)

        # Labels
        pos_label = Text("Position", color=WHITE, font_size=20)
        rate_label = Text("Rate of Change", color=WHITE, font_size=20)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_in_area(tree, 'D2', 'E2', scale_factor=0.6)
        self.place_in_area(graph_group, 'C3', 'F6', scale_factor=0.85)
        self.play(FadeIn(tree), FadeIn(graph_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_at_grid(pos_label, 'B2', scale_factor=0.7)
        self.play(Write(pos_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.place_at_grid(rate_label, 'B5', scale_factor=0.7)
        self.play(Write(rate_label))
        
        # Flashing colors for synthesis
        self.play(
            pos_label.animate.set_color(YELLOW),
            rate_label.animate.set_color(YELLOW),
            run_time=2
        )
        self.wait(2)
