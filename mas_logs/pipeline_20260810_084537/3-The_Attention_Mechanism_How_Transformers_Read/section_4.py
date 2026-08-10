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
            "Calculate similarity using a dot product.",
            "Apply Softmax to normalize the scores.",
            "The result is a heat map.",
            "Intensity reveals strength of word connections.",
            "High intensity signifies deep syntactic relationship."
        ]
        self.setup_layout("Mathematical Visualization (Softmax)", lecture_lines)
        
        # Color objects for lines
        colors = [YELLOW, BLUE, GREEN, RED, ORANGE]
        
        # === Animation for Lecture Line 1 ===
        vec = MathTex(r"\vec{x} = [x_1, x_2, x_3]").set_color(colors[0])
        self.place_at_grid(vec, 'B4', scale_factor=0.9)
        self.play(FadeIn(vec))
        self.lecture[0].set_color(colors[0])
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        softmax = MathTex(r"\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}").set_color(colors[1])
        self.place_at_grid(softmax, 'C4', scale_factor=0.8)
        self.play(ReplacementTransform(vec.copy(), softmax))
        self.lecture[1].set_color(colors[1])
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Heatmap represented by a grid of squares
        heatmap = VGroup(*[Square(side_length=0.7, fill_opacity=0.5, color=GREY).set_fill(color=BLUE_E) for _ in range(9)])
        heatmap.arrange_in_grid(3, 3, buff=0.1)
        # Load asset
        heatmap_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heatmap.svg")
        heatmap_group = VGroup(heatmap, heatmap_asset)
        
        self.place_in_area(heatmap_group, 'D3', 'F5', scale_factor=0.7)
        self.play(FadeIn(heatmap_group))
        self.lecture[2].set_color(colors[2])
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight values summing to 1 (cyan)
        cyan_highlight = Text("Sum = 1", color="#00FFFF", font_size=20)
        self.place_at_grid(cyan_highlight, 'E2', scale_factor=1.0)
        
        self.play(heatmap[0].animate.set_fill(color=RED, opacity=0.8),
                  heatmap[4].animate.set_fill(color=RED, opacity=0.8),
                  FadeIn(cyan_highlight))
        self.lecture[3].set_color(colors[3])
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        self.wait(1)
