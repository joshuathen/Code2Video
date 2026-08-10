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
        lecture_lines_raw = [
            "Facts exist across the distributed network.",
            "No single cell holds one specific fact.",
            "Holographic storage requires retraining to change."
        ]
        self.setup_layout("Conclusion: Distributed Representation", lecture_lines_raw)
        
        # Prepare lecture text
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines_raw]
        self.lecture_group = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(self.lecture_group, 'A1', 'C2', scale_factor=0.9)
        self.add(self.lecture_group)
        
        # Create Neurons
        neuron_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg"
        network = VGroup(*[
            SVGMobject(neuron_path, color=WHITE).scale(0.3) 
            for _ in range(16)
        ])
        
        # Grid placement for neurons
        positions = ["B3", "B4", "B5", "B6", "C3", "C4", "C5", "C6", "D3", "D4", "D5", "D6", "E3", "E4", "E5", "E6"]
        for i, pos in enumerate(positions):
            network[i].move_to(self.grid[pos])
            
        connections = VGroup(*[
            Line(network[i].get_center(), network[j].get_center(), color=GRAY, stroke_width=1)
            for i in range(len(network)) for j in range(i+1, len(network))
            if np.linalg.norm(network[i].get_center() - network[j].get_center()) < 1.5
        ])
        
        grid_visual = VGroup(connections, network)
        self.place_in_area(grid_visual, 'A3', 'F6', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(connections), FadeIn(network))
        self.lecture_group[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        subset = VGroup(*[network[i] for i in [0, 5, 10, 15]])
        self.play(subset.animate.set_color("#FF00FF"))
        self.lecture_group[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(connections), network.animate.set_color("#00FFFF"))
        self.lecture_group[2].set_color("#00FFFF")
        self.wait(2)
