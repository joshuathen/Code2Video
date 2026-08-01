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

class Section1Scene(TeachingScene):
    def construct(self):
        title = "The Digital Fingerprint: What is a Hash?"
        lines = [
            "A hash function transforms any data into fixed-length code.",
            "It acts like a digital fingerprint for your data.",
            "Like making juice, you can't reverse the process easily.",
            "Even a tiny change creates a completely new hash.",
            "This uniqueness makes it perfect for verifying data integrity."
        ]
        
        self.setup_layout(title, lines)
        
        # Asset paths
        cat_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png"
        machine_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg"

        # === Animation for Lecture Line 1 ===
        # Highlight current line
        self.lecture[0].set_color("#FFFF00")
        
        # [Visual Critic Fix: Issue 19] Use C2-E4 to avoid overlap
        input_box = Rectangle(width=3, height=2, color="#0000FF", stroke_width=4)
        self.place_in_area(input_box, "C2", "E4")
        input_label = Text("Input Box", font_size=20, color="#0000FF")
        input_label.next_to(input_box, UP, buff=0.1)
        
        self.play(Create(input_box), FadeIn(input_label))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Reset color and highlight next
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        # [Visual Critic Fix: Issue 20] Use C1 to avoid overlap
        # [Asset Integration: Issue 16] Load cat.png
        cat_icon = ImageMobject(cat_asset)
        self.place_at_grid(cat_icon, "C1", scale_factor=0.8)
        
        self.play(FadeIn(cat_icon))
        self.play(cat_icon.animate.move_to(input_box.get_center()))
        self.play(FadeOut(cat_icon))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        # [Visual Critic Fix: Issue 19] Use C2-E4
        # [Asset Integration: Issue 16] Load machine.svg
        hash_machine = SVGMobject(machine_asset)
        hash_machine.set_color("#808080")
        self.place_in_area(hash_machine, "C2", "E4", scale_factor=1.2)
        machine_label = Text("Hash Machine", font_size=22, color="#808080")
        machine_label.next_to(hash_machine, UP, buff=0.1)
        
        self.play(
            ReplacementTransform(input_box, hash_machine),
            ReplacementTransform(input_label, machine_label)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFF00")
        
        # Unique 64-character hex code
        hex_val = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        hex_display = hex_val[:32] + "\n" + hex_val[32:]
        # Use Text instead of MathTex (L022)
        hex_code = Text(hex_display, font_size=16, color="#00FF00", font="Courier New")
        
        # [Visual Critic Fix: Issue 21] Place in area F1-F6
        self.place_in_area(hex_code, "F1", "F6", scale_factor=0.8)
        
        # Animation: Processing and output
        self.play(Indicate(hash_machine, color="#00FF00")) # Use Indicate (L004)
        self.play(FadeIn(hex_code, shift=DOWN))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFF00")
        
        # [Visual Critic Fix: Issue 20] Modified cat at C1
        # [Asset Integration: Issue 16] Use cat.png again
        modified_cat = ImageMobject(cat_asset)
        self.place_at_grid(modified_cat, "C1", scale_factor=0.8)
        
        # Symbolize "change one pixel" with a small red dot
        pixel_change = Dot(color="#FF0000", radius=0.05)
        pixel_change.move_to(modified_cat.get_center() + 0.1 * RIGHT + 0.1 * UP)
        modified_group = Group(modified_cat, pixel_change)
        
        # New Hash for comparison
        new_hex_val = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        new_hex_display = new_hex_val[:32] + "\n" + new_hex_val[32:]
        new_hex_code = Text(new_hex_display, font_size=16, color="#FF0000", font="Courier New")
        self.place_in_area(new_hex_code, "F1", "F6", scale_factor=0.8)
        
        self.play(FadeIn(modified_group))
        self.play(modified_group.animate.move_to(hash_machine.get_center()))
        self.play(FadeOut(modified_group), Indicate(hash_machine, color="#FF0000"))
        
        # Flash the hash red then swap to show it's completely different
        self.play(hex_code.animate.set_color("#FF0000"))
        self.play(ReplacementTransform(hex_code, new_hex_code))
        
        self.wait(1.5)
        
        # Final highlight cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(1.5)
