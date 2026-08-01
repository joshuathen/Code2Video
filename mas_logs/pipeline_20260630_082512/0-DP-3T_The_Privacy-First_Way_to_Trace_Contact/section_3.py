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

class Section3Scene(TeachingScene):
    def construct(self):
        # Define Colors
        BOLT_COLOR = "#00CCFF"
        ACE_COLOR = "#FFCC00"
        RPI_COLOR = "#00FF00"
        ALERT_COLOR = "#FF0000"
        BG_DARK = "#222222"

        self.setup_layout(
            "Phase 1: The Secret Handshake (Broadcasting)", 
            [
                'Phones broadcast rotating identifiers every few minutes.', 
                'Nearby devices listen and store these random codes.', 
                'No identity or location is ever shared.', 
                'Codes change frequently to prevent long-term tracking.'
            ]
        )

        # Helper for creating a phone
        def create_phone(label_text, color):
            phone_body = RoundedRectangle(corner_radius=0.1, height=1.2, width=0.7, color=color, fill_opacity=0.2)
            screen = Rectangle(height=0.8, width=0.55, color=color, fill_opacity=0.1).move_to(phone_body.get_center() + UP * 0.1)
            button = Circle(radius=0.05, color=color).move_to(phone_body.get_center() + DOWN * 0.45)
            label = Text(label_text, font_size=18, color=color).next_to(phone_body, UP, buff=0.1)
            return VGroup(phone_body, screen, button, label)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BOLT_COLOR)
        bolt_phone = create_phone("Bolt", BOLT_COLOR)
        self.place_at_grid(bolt_phone, "B2")
        
        rpi_label = Text("RPI-1", font_size=16, color=RPI_COLOR)
        rpi_label.next_to(bolt_phone, DOWN, buff=0.2)
        
        # Waves
        waves = VGroup(*[
            Circle(radius=0.1, color=BOLT_COLOR, stroke_opacity=0.6)
            for _ in range(3)
        ]).move_to(bolt_phone.get_center())

        self.add(bolt_phone, rpi_label)
        self.play(FadeIn(bolt_phone), FadeIn(rpi_label))
        
        # Animate waves
        def update_wave(obj, dt):
            obj.scale(1 + 0.5 * dt)
            obj.set_stroke(opacity=max(0, obj.get_stroke_opacity() - 0.3 * dt))
            if obj.get_stroke_opacity() <= 0:
                obj.scale(0.01)
                obj.set_stroke(opacity=0.6)
                obj.move_to(bolt_phone.get_center())

        for i, wave in enumerate(waves):
            wave.scale(1 + i * 0.5)
            wave.set_stroke(opacity=0.6 - i * 0.2)
            wave.add_updater(update_wave)
        
        self.add(waves)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(ACE_COLOR)
        
        ace_phone = create_phone("Ace", ACE_COLOR)
        self.place_at_grid(ace_phone, "B5")
        
        # Diary
        diary_box = RoundedRectangle(corner_radius=0.05, height=1.5, width=1.5, color=GRAY_A, fill_opacity=0.1)
        self.place_at_grid(diary_box, "E5")
        diary_title = Text("Ace's Diary", font_size=14, color=WHITE).next_to(diary_box, UP, buff=0.1)
        diary_entry = Text("RPI-1", font_size=14, color=RPI_COLOR).move_to(diary_box.get_top() + DOWN * 0.3)
        diary_group = VGroup(diary_box, diary_title)

        self.play(FadeIn(ace_phone), FadeIn(diary_group))
        
        # Flying RPI packet
        packet = Text("RPI-1", font_size=14, color=RPI_COLOR).move_to(bolt_phone.get_center())
        self.play(packet.animate.move_to(ace_phone.get_center()), run_time=1.5)
        self.play(packet.animate.move_to(diary_entry.get_center()), FadeIn(diary_entry))
        self.remove(packet)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ALERT_COLOR)
        
        id_label = Text("ID: Bolt", font_size=18, color=WHITE)
        loc_label = Text("Location: 40.7°N", font_size=18, color=WHITE)
        privacy_vgroup = VGroup(id_label, loc_label).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(privacy_vgroup, "D2")
        
        cross = Cross(privacy_vgroup, stroke_color=ALERT_COLOR, stroke_width=8)
        
        self.play(FadeIn(privacy_vgroup))
        self.play(Create(cross))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RPI_COLOR)
        
        # Cycle RPI codes
        rpi_codes = ["RPI-2", "RPI-3", "RPI-4"]
        
        for code in rpi_codes:
            new_rpi = Text(code, font_size=16, color=RPI_COLOR).move_to(rpi_label.get_center())
            new_diary_entry = Text(code, font_size=14, color=RPI_COLOR).next_to(diary_entry, DOWN, buff=0.1)
            
            self.play(
                Transform(rpi_label, new_rpi),
                run_time=0.8
            )
            
            packet = Text(code, font_size=14, color=RPI_COLOR).move_to(bolt_phone.get_center())
            self.play(
                packet.animate.move_to(ace_phone.get_center()),
                run_time=0.6
            )
            self.play(
                packet.animate.move_to(new_diary_entry.get_center()),
                FadeIn(new_diary_entry),
                run_time=0.4
            )
            self.remove(packet)
            diary_entry = new_diary_entry # shift reference for next entry
        
        self.wait(2)
        
        # Cleanup updaters
        for wave in waves:
            wave.remove_updater(update_wave)
        
        self.lecture[3].set_color(WHITE)
        self.wait(1)
