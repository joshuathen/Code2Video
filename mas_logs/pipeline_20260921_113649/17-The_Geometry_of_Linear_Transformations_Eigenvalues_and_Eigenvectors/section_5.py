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
        self.setup_layout("Real-World Application Summary", [
            "Eigenvalues and vectors are used in stability analysis.",
            "They enable powerful tools like image compression.",
            "They are fundamental to modern quantum mechanics."
        ])
        
        # Animations
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg]
        network = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg", color=WHITE).scale(0.2) for _ in range(10)])
        network.arrange_in_grid(2, 5, buff=0.2)
        # Applying layout fixes per Critic instructions
        self.place_in_area(network, 'B3', 'E6', scale_factor=0.7)
        
        network_label = Text("Network", color=WHITE, font_size=20)
        self.place_at_grid(network_label, 'B2', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(network), Write(network_label))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Highlight specific nodes (the svg objects)
        key_nodes = VGroup(*[network[i] for i in [0, 4, 7, 9]])
        self.play(key_nodes.animate.set_color("#00FF00"))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/atom.svg]
        rank_icons = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/atom.svg", color="#FFFF00").scale(0.15) for _ in range(4)])
        rank_labels = VGroup(*[Text("Rank", color="#FFFF00", font_size=14) for _ in range(4)])
        
        rank_info = VGroup()
        for i, node in enumerate(key_nodes):
            pair = VGroup(rank_icons[i], rank_labels[i]).arrange(RIGHT, buff=0.1)
            pair.next_to(node, UP, buff=0.1)
            rank_info.add(pair)
        
        self.play(FadeIn(rank_info))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
