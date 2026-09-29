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
            "DP-3T ensures privacy through decentralized storage.",
            "No central database records individual user encounters.",
            "Anonymity is maintained by design and cryptography."
        ]
        self.setup_layout("Privacy Guarantees and Conclusion", lecture_lines)
        
        # Elements
        # Using SVG asset for padlock as requested
        padlock = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/padlock.svg", color=WHITE)
        zero_knowledge = Text("Anonymity", font_size=36, color="#00FFFF")
        box = SurroundingRectangle(zero_knowledge, color="#00FFFF", buff=0.2)
        zk_group = VGroup(box, zero_knowledge)
        privacy_text = Text("Privacy Guaranteed", font_size=48, color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        # Visual: Scattered nodes
        nodes = VGroup(*[Dot(self.grid[pos], radius=0.1) for pos in ['B3', 'B5', 'C4', 'D3', 'D5']])
        connections = VGroup(*[Line(nodes[i].get_center(), nodes[j].get_center(), color=GRAY) for i in range(len(nodes)) for j in range(i+1, len(nodes))])
        decentralized_web = VGroup(connections, nodes)
        
        self.play(FadeIn(decentralized_web))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        central_db = Text("Central Database", font_size=30, color=WHITE)
        self.place_at_grid(central_db, 'C2', scale_factor=0.8)
        cross_line = Line(start=central_db.get_left(), end=central_db.get_right(), color="#FF0000", stroke_width=6)
        
        self.play(FadeIn(central_db))
        self.play(Create(cross_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.place_at_grid(padlock, 'B5', scale_factor=0.8)
        self.place_at_grid(zk_group, 'C5', scale_factor=0.7)
        
        self.play(FadeIn(padlock), FadeIn(zk_group))
        self.wait(2)
