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
        self.setup_layout("Scaling Up: From Transformer to LLM", 
                          ["Scaling stacks many Transformer layers together.", 
                           "Deep stacking captures subtle logic and nuance.", 
                           "LLMs represent vast networks of knowledge."])
        
        # === Animation for Lecture Line 1 ===
        # Create multiple rectangular layers stacked vertically using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg]
        server_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg", color="#FFFFFF")
        
        layers = VGroup()
        for i in range(5):
            layer = server_icon.copy()
            # Fixed: Calculate row index by converting 'E' to index and subtracting i, then remapping to char
            row_char = chr(ord('E') - i)
            self.place_at_grid(layer, f"{row_char}3", scale_factor=0.5)
            layers.add(layer)
        
        self.play(Create(layers))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Pulse each layer to show depth and nuance being captured.
        pulses = VGroup()
        for layer in layers:
            pulse = layer.copy().set_color("#0000FF")
            pulses.add(pulse)
        
        self.play(Flash(layers, color="#0000FF", line_length=0.1, num_lines=10))
        self.lecture[1].set_color("#0000FF")

        # === Animation for Lecture Line 3 ===
        # Expand the stack to show the vast network of knowledge using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/network.svg]
        network_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/network.svg", color="#FF00FF")
        self.place_in_area(network_icon, "B2", "D5", scale_factor=1.2)
        
        self.play(ReplacementTransform(layers, network_icon))
        self.lecture[2].set_color("#FF00FF")
