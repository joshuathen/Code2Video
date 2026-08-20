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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Quantifying the Mistake: Loss Function", ["Loss function quantifies our mistake.", "Visualize it as a bowl landscape.", "We aim for the lowest point."])
        
        # === Animation for Lecture Line 1 ===
        # Show ground truth and prediction side by side.
        lbl_truth = Text("Truth: Cat", color=BLUE, font_size=24)
        lbl_pred = Text("Prediction: Dog", color=YELLOW, font_size=24)
        self.place_at_grid(lbl_truth, "B2", scale_factor=0.8)
        self.place_at_grid(lbl_pred, "B5", scale_factor=0.8)
        self.play(Write(lbl_truth), Write(lbl_pred))
        self.lecture[0].set_color(BLUE)
        
        # Animate difference bar appearing in #FF0000.
        diff_bar = Rectangle(height=0.5, width=2.0, color=RED, fill_opacity=0.5)
        self.place_at_grid(diff_bar, "C4", scale_factor=0.8)
        self.play(FadeIn(diff_bar))

        # === Animation for Lecture Line 2 ===
        # Visualize it as a bowl landscape.
        self.lecture[1].set_color(GREEN)
        
        # Load bowl asset
        bowl_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bowl.svg", color=WHITE)
        self.place_in_area(bowl_img, "C2", "E5", scale_factor=0.6)
        
        # Function graph bowl
        bowl_graph = FunctionGraph(lambda x: x**2, x_range=[-1.5, 1.5], color=WHITE)
        self.place_in_area(bowl_graph, "C2", "E5", scale_factor=0.6)
        
        self.play(FadeIn(bowl_img), Create(bowl_graph))

        # === Animation for Lecture Line 3 ===
        # We aim for the lowest point.
        self.lecture[2].set_color(YELLOW)
        dot = Dot(color=YELLOW)
        self.place_at_grid(dot, "E3", scale_factor=0.8)
        self.play(FadeIn(dot))
        self.play(dot.animate.move_to(bowl_graph.point_from_proportion(0.5)))
        self.wait(1)
