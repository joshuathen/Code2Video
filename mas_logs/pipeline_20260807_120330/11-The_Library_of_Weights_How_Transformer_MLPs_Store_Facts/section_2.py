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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup the scene layout
        title = "Anatomy Check: Attention vs. MLP"
        lines = [
            "A Transformer block has two main components.",
            "Attention connects words, while the MLP stores facts.",
            "Think of the MLP as a massive filing cabinet."
        ]
        self.setup_layout(title, lines)
        
        # Define Colors
        BLOCK_COLOR = "#ADD8E6"  # Light Blue
        ATTN_COLOR = "#FFA500"   # Orange
        MLP_COLOR = "#90EE90"    # Light Green
        GREY_COLOR = "#808080"

        # === Animation for Lecture Line 1 ===
        # Display a 'Transformer Layer' block with 'Attention' and 'MLP' sub-blocks.
        self.lecture[0].set_color(BLOCK_COLOR)
        
        # Main container for the Transformer block
        transformer_block = Rectangle(width=4.5, height=5.2, color=BLOCK_COLOR, stroke_width=2)
        self.place_in_area(transformer_block, "A2", "F5")
        
        # Attention sub-block (Top half)
        attn_box = RoundedRectangle(corner_radius=0.1, width=4.0, height=2.0, color=BLOCK_COLOR, fill_opacity=0.1)
        self.place_in_area(attn_box, "B2", "C5")
        attn_label = Text("Attention", font_size=24, color=BLOCK_COLOR)
        # Resolved Issue 35: Reposition to avoid vertical crowding
        self.place_in_area(attn_label, "B2", "B5", scale_factor=0.8)

        # MLP sub-block (Bottom half)
        mlp_box = RoundedRectangle(corner_radius=0.1, width=4.0, height=2.0, color=BLOCK_COLOR, fill_opacity=0.1)
        self.place_in_area(mlp_box, "D2", "E5")
        mlp_label = Text("MLP", font_size=24, color=BLOCK_COLOR)
        # Resolved Issue 34: Resize and center to avoid excessive height scaling
        self.place_in_area(mlp_label, "D3", "D4", scale_factor=0.8)

        self.play(Create(transformer_block))
        self.play(
            AnimationGroup(
                Create(attn_box),
                Write(attn_label),
                Create(mlp_box),
                Write(mlp_label),
                lag_ratio=0.3
            )
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the 'Attention' block; show moving lines connecting sample words.
        self.play(
            self.lecture[0].animate.set_color(GREY_COLOR),
            self.lecture[1].animate.set_color(ATTN_COLOR),
            attn_box.animate.set_stroke(color=ATTN_COLOR, width=4),
            attn_label.animate.set_color(ATTN_COLOR)
        )

        # Words inside Attention box
        words = VGroup(
            Text("The", font_size=20),
            Text("cat", font_size=20),
            Text("sat", font_size=20)
        )
        self.place_at_grid(words[0], "C2", scale_factor=1)
        self.place_in_area(words[1], "C3", "C4", scale_factor=1)
        self.place_at_grid(words[2], "C5", scale_factor=1)

        # Curved lines (arcs) representing attention connections
        arc1 = ArcBetweenPoints(words[0].get_top() + UP*0.1, words[1].get_top() + UP*0.1, angle=-TAU/4, color=ATTN_COLOR, stroke_width=2)
        arc2 = ArcBetweenPoints(words[1].get_top() + UP*0.1, words[2].get_top() + UP*0.1, angle=-TAU/4, color=ATTN_COLOR, stroke_width=2)
        arc3 = ArcBetweenPoints(words[0].get_top() + UP*0.2, words[2].get_top() + UP*0.2, angle=-TAU/3, color=ATTN_COLOR, stroke_width=2)

        self.play(FadeIn(words))
        self.play(Create(arc1), Create(arc2), Create(arc3), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the 'MLP' block; show a 'Filing Cabinet' icon inside it.
        self.play(
            self.lecture[1].animate.set_color(GREY_COLOR),
            self.lecture[2].animate.set_color(MLP_COLOR),
            attn_box.animate.set_stroke(color=BLOCK_COLOR, width=2),
            attn_label.animate.set_color(BLOCK_COLOR),
            mlp_box.animate.set_stroke(color=MLP_COLOR, width=4),
            mlp_label.animate.set_color(MLP_COLOR),
            FadeOut(words), FadeOut(arc1), FadeOut(arc2), FadeOut(arc3)
        )

        # Resolved Issue 30: Use SVG Asset for Filing Cabinet
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cabinet.svg]
        cabinet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cabinet.svg")
        cabinet.set_color(MLP_COLOR)
        
        # Resolved Issue 36: Resize cabinet to avoid cramped diagram
        self.place_in_area(cabinet, "E2", "E5", scale_factor=0.75)
        
        self.play(DrawBorderThenFill(cabinet))
        self.wait(2)
