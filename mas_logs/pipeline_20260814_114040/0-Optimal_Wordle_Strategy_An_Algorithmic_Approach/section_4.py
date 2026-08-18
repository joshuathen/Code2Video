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
        self.setup_layout("Algorithmic Walkthrough: The Minimax Algorithm", [
            "Minimax minimizes the maximum remaining possibilities.",
            "It guarantees worst-case performance bounds.",
            "Decision trees visualize this branching process."
        ])
        
        # --- Assets ---
        chess_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chess.svg")
        pawn_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pawn.svg")
        game_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/game.svg")
        king_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/king.svg")
        
        # --- Visualization Setup ---
        root = Circle(radius=0.3, color=WHITE, fill_opacity=0.5)
        self.place_at_grid(root, 'C4')
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        title_tag = Text("Minimax Algorithm", font_size=32, color="#FFFFFF")
        self.place_at_grid(title_tag, 'A5')
        self.place_at_grid(chess_icon, 'B5', scale_factor=0.3)
        self.play(Write(title_tag), Create(root), FadeIn(chess_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.place_at_grid(pawn_icon, 'B3', scale_factor=0.3)
        
        # Branching nodes
        nodes = VGroup()
        for i in range(3):
            n = Circle(radius=0.2, color="#FFFF00", fill_opacity=0.5)
            nodes.add(n)
        
        self.place_at_grid(nodes[0], 'D3')
        self.place_at_grid(nodes[1], 'D4')
        self.place_at_grid(nodes[2], 'D5')
        
        lines = VGroup(*[Line(root.get_center(), n.get_center(), stroke_width=2) for n in nodes])
        self.play(FadeIn(pawn_icon), Create(lines), Create(nodes))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.place_at_grid(game_icon, 'C2', scale_factor=0.3)
        
        terminal = Circle(radius=0.2, color="#00FF00", fill_opacity=1)
        self.place_at_grid(terminal, 'E5', scale_factor=0.9)
        target = Dot(color="#00FF00").move_to(terminal.get_center())
        
        self.play(FadeIn(game_icon), Create(terminal), FadeIn(target))
        
        # Propagation animation
        val_text = Text("Value", font_size=16, color="#FF0000")
        self.place_at_grid(val_text, 'F5', scale_factor=0.8)
        self.play(Write(val_text))
        self.play(val_text.animate.move_to(root.get_center()), run_time=1.5)
        
        # Highlight best choice
        self.place_at_grid(king_icon, 'B4', scale_factor=0.3)
        self.play(FadeIn(king_icon))
        
        self.wait(1)
